// vggt_spectra.cpp -- SPECTRA stages for the PipeGen-generated DeepStream pipeline.
//
// PipeGen v0.5.1 generated the pipeline (nvurisrcbin -> nvstreammux -> nvinfer_backport
// with the PipeGen-built FP16 VGGT engine -> nvcustomprocess -> meta_json -> JPEG frames)
// but its repair loop left the VGGT decode stage as a pass-through stub and fed one
// stretched frame per inference. This one library provides both missing stages for the
// `nvcustomprocess` element (CPU mode); it is loaded twice with different entry points
// and dlopen shares its state between them:
//
//   VggtPair   (before nvinfer)  writes the VGGT two-view atlas [frame t-1s | frame t]
//                                into the 1036x518 surface nvinfer consumes.
//   VggtDecode (after nvinfer)   pose_enc/depth/depth_conf -> cameras -> world points ->
//                                confidence / voxel / statistical-outlier filtering ->
//                                PLY + metadata.json + geometry_ready event, and draws
//                                [current view | depth] onto the output frame.
//
// It mirrors the Python reference (spectra/contract.py, spectra/reconstruction.py,
// vggt/utils/{load_fn,pose_enc,rotation,geometry}.py). Differences: OpenCV INTER_AREA
// stands in for PIL's antialiased bicubic resize, the model sees the view after an
// NV12 (4:2:0) round trip, and the outlier filter uses an approximate FLANN k-d tree.
//
// CONFIGURATION (env vars -- nvcustomprocess has no config-file property)
//   VGGT_OUT              output directory              (/home/atharv/spectra/output)
//   VGGT_SRC_W, VGGT_SRC_H  source video size, for aspect (778, 520)
//   VGGT_FPS              source frame rate              (4)
//   VGGT_SAMPLE_SECONDS   time between the two views     (1.0, as Config.sample_seconds)
//   VGGT_ENGINE           engine path, recorded in outputs

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgproc.hpp>

#include "nvcustomprocess.h"
#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"

namespace fs = std::filesystem;

namespace {

constexpr int S = 518;                       // VGGT view size (contract.SIZE)
constexpr double CONF_PERCENTILE = 50.0;     // Config.confidence_percentile
constexpr size_t MAX_POINTS = 100000;        // Config.max_points
constexpr double VOXEL = 0.005;              // Config.voxel_size
constexpr int OUTLIER_K = 20;                // Config.outlier_neighbors
constexpr double OUTLIER_STD = 2.0;          // Config.outlier_std_ratio

struct Settings {
    std::string out = "/home/atharv/spectra/output", engine;
    int src_w = 778, src_h = 520, stride = 4;
    double fps = 4.0;
};

struct Pair { int first = -1, second = -1; };

std::mutex g_mu;
Settings g_cfg;
bool g_init = false;
std::map<int, cv::Mat> g_views;   // frame_num -> padded 518x518 BGR view (pre-NV12)
std::map<int, Pair> g_pairs;      // frame_num -> views packed into that frame's atlas
bool g_wrote_primary = false;
int g_pairs_written = 0;
std::chrono::steady_clock::time_point g_t0, g_last;

const char* env(const char* k, const char* d) { const char* v = getenv(k); return v && *v ? v : d; }

void init_locked() {
    if (g_init) return;
    g_cfg.out = env("VGGT_OUT", g_cfg.out.c_str());
    g_cfg.engine = env("VGGT_ENGINE", "");
    g_cfg.src_w = atoi(env("VGGT_SRC_W", "778"));
    g_cfg.src_h = atoi(env("VGGT_SRC_H", "520"));
    g_cfg.fps = atof(env("VGGT_FPS", "4"));
    double sample = atof(env("VGGT_SAMPLE_SECONDS", "1.0"));
    g_cfg.stride = std::max(1, (int) std::nearbyint(g_cfg.fps * sample));   // inputs.py stride
    fs::create_directories(fs::path(g_cfg.out) / "pairs");
    fs::remove(fs::path(g_cfg.out) / "events.jsonl");
    g_t0 = g_last = std::chrono::steady_clock::now();
    g_init = true;
    fprintf(stderr, "[vggt] out=%s stride=%d frames (%.2fs at %.2f fps)\n",
            g_cfg.out.c_str(), g_cfg.stride, sample, g_cfg.fps);
}

int frame_num(NvDsCustomPostProcessFrameInCpu* in) {
    auto* fm = (NvDsFrameMeta*) in->frame_meta;
    return fm ? (int) fm->frame_num : -1;
}

cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu* in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r) memcpy(yuv.ptr(r), in->Y_host + (size_t) r * in->Y_pitch, W);
    for (int r = 0; r < H / 2; ++r) memcpy(yuv.ptr(H + r), in->UV_host + (size_t) r * in->UV_pitch, W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

void write_nv12(const cv::Mat& bgr, NvDsCustomPostProcessFrameInCpu* in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t* y = i420.data;
    const uint8_t* u = y + (size_t) W * H;
    const uint8_t* v = u + (size_t) (W / 2) * (H / 2);
    for (int r = 0; r < H; ++r) memcpy(in->Y_host + (size_t) r * in->Y_pitch, y + (size_t) r * W, W);
    for (int r = 0; r < H / 2; ++r) {
        uint8_t* d = in->UV_host + (size_t) r * in->UV_pitch;
        for (int c = 0; c < W / 2; ++c) { d[2*c] = u[r*(W/2)+c]; d[2*c+1] = v[r*(W/2)+c]; }
    }
}

// load_fn.load_and_preprocess_images(mode='pad'): longest side -> 518, other side
// rounded to a multiple of 14, centred on a white 518x518 canvas. nvstreammux scaled
// the source to the surface size, so resize from the surface back to source aspect.
cv::Mat make_view(const cv::Mat& surface) {
    int nw, nh;
    if (g_cfg.src_w >= g_cfg.src_h) {
        nw = S; nh = (int) std::nearbyint(g_cfg.src_h * ((double) S / g_cfg.src_w) / 14.0) * 14;
    } else {
        nh = S; nw = (int) std::nearbyint(g_cfg.src_w * ((double) S / g_cfg.src_h) / 14.0) * 14;
    }
    cv::Mat resized; cv::resize(surface, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    cv::Mat view(S, S, CV_8UC3, cv::Scalar(255, 255, 255));
    resized.copyTo(view(cv::Rect((S - nw) / 2, (S - nh) / 2, nw, nh)));
    return view;
}

// ---- tensors ----------------------------------------------------------------
struct Tensor { std::vector<float> v; std::vector<int> dims; };

bool get_tensor(NvDsCustomPostProcessFrameInCpu* in, const char* name, Tensor& t) {
    for (size_t mi = 0; mi < in->num_tensor_metas; ++mi) {
        auto* tm = (NvDsInferTensorMeta*) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            NvDsInferLayerInfo& info = tm->output_layers_info[li];
            if (!info.layerName || strcmp(info.layerName, name) != 0 || !tm->out_buf_ptrs_host[li]) continue;
            size_t n = 1; t.dims.clear();
            for (unsigned d = 0; d < info.inferDims.numDims; ++d) { t.dims.push_back(info.inferDims.d[d]); n *= info.inferDims.d[d]; }
            t.v.resize(n);
            if (info.dataType == FLOAT) {
                memcpy(t.v.data(), tm->out_buf_ptrs_host[li], n * sizeof(float));
            } else if (info.dataType == HALF) {
                cv::Mat h(1, (int) n, CV_16F, tm->out_buf_ptrs_host[li]), f;
                h.convertTo(f, CV_32F); memcpy(t.v.data(), f.ptr<float>(), n * sizeof(float));
            } else {
                fprintf(stderr, "[vggt] %s has unsupported data type %d\n", name, (int) info.dataType);
                return false;
            }
            return true;
        }
    }
    return false;
}

// ---- geometry ---------------------------------------------------------------
struct Camera { double R[9], t[3], K[9]; };

// pose_enc.pose_encoding_to_extri_intri + rotation.quat_to_mat (XYZW, scalar last)
bool decode_camera(const float* p, Camera& c) {
    double i = p[3], j = p[4], k = p[5], r = p[6];
    double qq = i*i + j*j + k*k + r*r;
    if (std::sqrt(qq) < 1e-8) return false;
    double s = 2.0 / qq;
    double R[9] = {1 - s*(j*j + k*k), s*(i*j - k*r),     s*(i*k + j*r),
                   s*(i*j + k*r),     1 - s*(i*i + k*k), s*(j*k - i*r),
                   s*(i*k - j*r),     s*(j*k + i*r),     1 - s*(i*i + j*j)};
    memcpy(c.R, R, sizeof R);
    for (int a = 0; a < 3; ++a) c.t[a] = p[a];
    double fy = (S / 2.0) / std::tan(p[7] / 2.0), fx = (S / 2.0) / std::tan(p[8] / 2.0);
    double K[9] = {fx, 0, S / 2.0, 0, fy, S / 2.0, 0, 0, 1};
    memcpy(c.K, K, sizeof K);
    for (double x : c.R) if (!std::isfinite(x)) return false;
    for (double x : c.K) if (!std::isfinite(x)) return false;
    return std::isfinite(c.t[0]) && std::isfinite(c.t[1]) && std::isfinite(c.t[2]);
}

struct Cloud { std::vector<cv::Point3f> p; std::vector<cv::Point3f> c; };   // colours RGB in [0,1]

double percentile(std::vector<float> v, double q) {        // numpy default (linear)
    std::sort(v.begin(), v.end());
    double pos = (v.size() - 1) * q / 100.0;
    size_t lo = (size_t) std::floor(pos), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

// open3d PointCloud::VoxelDownSample: average points and colours per voxel
Cloud voxel_down(const Cloud& in, double voxel) {
    cv::Point3f mn(FLT_MAX, FLT_MAX, FLT_MAX);
    for (auto& p : in.p) { mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z); }
    double ox = mn.x - voxel * 0.5, oy = mn.y - voxel * 0.5, oz = mn.z - voxel * 0.5;
    struct Acc { double p[3] = {0,0,0}, c[3] = {0,0,0}; int n = 0; };
    std::unordered_map<uint64_t, size_t> index; std::vector<Acc> acc;
    for (size_t i = 0; i < in.p.size(); ++i) {
        auto& p = in.p[i];
        uint64_t kx = (uint64_t) std::floor((p.x - ox) / voxel), ky = (uint64_t) std::floor((p.y - oy) / voxel),
                 kz = (uint64_t) std::floor((p.z - oz) / voxel);
        uint64_t key = (kx & 0x1FFFFF) | ((ky & 0x1FFFFF) << 21) | ((kz & 0x1FFFFF) << 42);
        auto it = index.find(key);
        size_t slot = it == index.end() ? (index[key] = acc.size(), acc.emplace_back(), acc.size() - 1) : it->second;
        Acc& a = acc[slot];
        a.p[0] += p.x; a.p[1] += p.y; a.p[2] += p.z;
        a.c[0] += in.c[i].x; a.c[1] += in.c[i].y; a.c[2] += in.c[i].z; a.n++;
    }
    Cloud out;
    for (auto& a : acc) {
        out.p.emplace_back(a.p[0] / a.n, a.p[1] / a.n, a.p[2] / a.n);
        out.c.emplace_back(a.c[0] / a.n, a.c[1] / a.n, a.c[2] / a.n);
    }
    return out;
}

// open3d PointCloud::RemoveStatisticalOutliers: mean distance to k nearest neighbours
// (the query point included), keep points within mean + ratio * sample std.
Cloud remove_outliers(const Cloud& in, int k, double ratio) {
    cv::Mat data((int) in.p.size(), 3, CV_32F, (void*) in.p.data());
    cv::flann::Index tree(data, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);
    cv::Mat idx, dist;
    tree.knnSearch(data, idx, dist, k, cv::flann::SearchParams(128));
    std::vector<double> avg(in.p.size());
    double sum = 0;
    for (int i = 0; i < data.rows; ++i) {
        double s = 0;
        for (int j = 0; j < k; ++j) s += std::sqrt(std::max(0.f, dist.at<float>(i, j)));
        avg[i] = s / k; sum += avg[i];
    }
    double mean = sum / avg.size(), sq = 0;
    for (double a : avg) sq += (a - mean) * (a - mean);
    double thr = mean + ratio * std::sqrt(sq / (avg.size() - 1));
    Cloud out;
    for (size_t i = 0; i < avg.size(); ++i)
        if (avg[i] <= thr) { out.p.push_back(in.p[i]); out.c.push_back(in.c[i]); }
    return out;
}

bool write_ply(const fs::path& path, const Cloud& cl) {
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\ncomment SPECTRA VGGT-1B FP16 TensorRT via DeepStream\n"
      << "element vertex " << cl.p.size() << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < cl.p.size(); ++i) {
        f.write((const char*) &cl.p[i], 12);
        uint8_t rgb[3] = {(uint8_t) std::lround(std::clamp(cl.c[i].x, 0.f, 1.f) * 255),
                          (uint8_t) std::lround(std::clamp(cl.c[i].y, 0.f, 1.f) * 255),
                          (uint8_t) std::lround(std::clamp(cl.c[i].z, 0.f, 1.f) * 255)};
        f.write((const char*) rgb, 3);
    }
    return (bool) f;
}

// Re-read the PLY and check the vertex count and that every coordinate is finite.
bool verify_ply(const fs::path& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    std::string line; size_t n = 0;
    while (std::getline(f, line) && line != "end_header")
        if (line.rfind("element vertex ", 0) == 0) n = std::stoull(line.substr(15));
    if (n != expected) return false;
    std::vector<char> rec(15);
    for (size_t i = 0; i < n; ++i) {
        if (!f.read(rec.data(), 15)) return false;
        float xyz[3]; memcpy(xyz, rec.data(), 12);
        if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) return false;
    }
    return true;
}

std::string jnum(double x) { char b[64]; snprintf(b, sizeof b, "%.9g", x); return b; }
std::string jmat(const double* m, int rows, int cols) {
    std::string s = "[";
    for (int r = 0; r < rows; ++r) {
        s += r ? ", [" : "[";
        for (int c = 0; c < cols; ++c) s += (c ? ", " : "") + jnum(m[r * cols + c]);
        s += "]";
    }
    return s + "]";
}

struct Result { size_t points = 0, input_points = 0; double threshold = 0; cv::Point3f mn, mx; };

// spectra/reconstruction.py export_predictions
bool reconstruct(const Tensor& pose, const Tensor& depth, const Tensor& conf, const cv::Mat views[2],
                 Camera cams[2], Cloud& cloud, Result& res) {
    for (int v = 0; v < 2; ++v)
        if (!decode_camera(&pose.v[v * 9], cams[v])) { fprintf(stderr, "[vggt] degenerate camera %d\n", v); return false; }
    std::vector<cv::Point3f> pts, cols; std::vector<float> cf; std::vector<char> valid;
    const size_t per = (size_t) S * S;
    for (int v = 0; v < 2; ++v) {
        const Camera& c = cams[v];
        for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
            size_t o = v * per + (size_t) y * S + x;
            float d = depth.v[o];
            // geometry.depth_to_cam_coords_points then cam->world with the inverse extrinsic
            double xc = (x - c.K[2]) * d / c.K[0], yc = (y - c.K[5]) * d / c.K[4], zc = d;
            double q[3] = {xc - c.t[0], yc - c.t[1], zc - c.t[2]}, w[3];
            for (int a = 0; a < 3; ++a) w[a] = c.R[0*3+a] * q[0] + c.R[1*3+a] * q[1] + c.R[2*3+a] * q[2];
            const cv::Vec3b& bgr = views[v].at<cv::Vec3b>(y, x);
            pts.emplace_back((float) w[0], (float) w[1], (float) w[2]);
            cols.emplace_back(bgr[2] / 255.f, bgr[1] / 255.f, bgr[0] / 255.f);
            cf.push_back(conf.v[o]);
            valid.push_back(std::isfinite(w[0]) && std::isfinite(w[1]) && std::isfinite(w[2]) &&
                            std::isfinite(conf.v[o]) && std::isfinite(d) && d > 0);
        }
    }
    res.input_points = pts.size();
    std::vector<float> vc;
    for (size_t i = 0; i < cf.size(); ++i) if (valid[i]) vc.push_back(cf[i]);
    if (vc.empty()) { fprintf(stderr, "[vggt] no finite positive-depth points\n"); return false; }
    res.threshold = percentile(vc, CONF_PERCENTILE);
    std::vector<size_t> keep;
    for (size_t i = 0; i < cf.size(); ++i) if (valid[i] && cf[i] >= res.threshold) keep.push_back(i);
    if (keep.size() > MAX_POINTS) {                     // np.linspace(0, n-1, max, dtype=int)
        std::vector<size_t> sub(MAX_POINTS);
        for (size_t k = 0; k < MAX_POINTS; ++k)
            sub[k] = keep[(size_t) ((double) k * (keep.size() - 1) / (MAX_POINTS - 1))];
        keep.swap(sub);
    }
    Cloud raw;
    for (size_t i : keep) { raw.p.push_back(pts[i]); raw.c.push_back(cols[i]); }
    cloud = voxel_down(raw, VOXEL);
    if ((int) cloud.p.size() > OUTLIER_K + 1) cloud = remove_outliers(cloud, OUTLIER_K, OUTLIER_STD);
    if (cloud.p.empty()) { fprintf(stderr, "[vggt] no points survived filtering\n"); return false; }
    res.points = cloud.p.size();
    res.mn = res.mx = cloud.p[0];
    for (auto& p : cloud.p) {
        res.mn.x = std::min(res.mn.x, p.x); res.mn.y = std::min(res.mn.y, p.y); res.mn.z = std::min(res.mn.z, p.z);
        res.mx.x = std::max(res.mx.x, p.x); res.mx.y = std::max(res.mx.y, p.y); res.mx.z = std::max(res.mx.z, p.z);
    }
    return true;
}

std::string metadata_json(const Pair& pr, const Camera cams[2], const Result& r, const std::string& ply,
                          double frame_seconds) {
    std::ostringstream m;
    m << "{\n  \"schema_version\": 1,\n  \"engine\": \"VGGT\",\n"
      << "  \"runtime\": {\"backend\": \"tensorrt\", \"host\": \"DeepStream 7.1 nvinfer_backport (PipeGen-generated pipeline)\", "
      << "\"precision\": \"fp16\", \"engine\": \"" << g_cfg.engine << "\"},\n"
      << "  \"scale\": \"arbitrary; uncalibrated\",\n"
      << "  \"coordinate_convention\": \"OpenCV camera: +X right, +Y down, +Z forward; extrinsics camera_from_world\",\n"
      << "  \"point_source\": \"VGGT depth unprojected with VGGT intrinsics/extrinsics\",\n"
      << "  \"point_count\": " << r.points << ",\n  \"input_point_count\": " << r.input_points << ",\n"
      << "  \"confidence_threshold\": " << jnum(r.threshold) << ",\n"
      << "  \"filters\": {\"confidence_percentile\": " << CONF_PERCENTILE << ", \"max_points\": " << MAX_POINTS
      << ", \"voxel_size\": " << VOXEL << ", \"outlier_neighbors\": " << OUTLIER_K << ", \"outlier_std_ratio\": " << OUTLIER_STD << "},\n"
      << "  \"cameras\": [";
    int frames[2] = {pr.first, pr.second};
    for (int v = 0; v < 2; ++v) {
        const Camera& c = cams[v];
        double ext[12], wfc[16] = {0}, center[3];
        for (int r2 = 0; r2 < 3; ++r2) { for (int k = 0; k < 3; ++k) ext[r2*4+k] = c.R[r2*3+k]; ext[r2*4+3] = c.t[r2]; }
        for (int r2 = 0; r2 < 3; ++r2) {                // [R^T | -R^T t]
            for (int k = 0; k < 3; ++k) wfc[r2*4+k] = c.R[k*3+r2];
            center[r2] = -(c.R[0*3+r2] * c.t[0] + c.R[1*3+r2] * c.t[1] + c.R[2*3+r2] * c.t[2]);
            wfc[r2*4+3] = center[r2];
        }
        wfc[15] = 1;
        m << (v ? ",\n    " : "\n    ") << "{\"frame_index\": " << frames[v]
          << ", \"timestamp_seconds\": " << jnum(frames[v] / g_cfg.fps)
          << ", \"camera_from_world\": " << jmat(ext, 3, 4) << ", \"intrinsic\": " << jmat(c.K, 3, 3)
          << ", \"world_from_camera\": " << jmat(wfc, 4, 4)
          << ", \"center\": [" << jnum(center[0]) << ", " << jnum(center[1]) << ", " << jnum(center[2]) << "]}";
    }
    m << "\n  ],\n  \"ply\": \"" << ply << "\",\n  \"ply_roundtrip_verified\": true,\n"
      << "  \"frame_wall_seconds\": " << jnum(frame_seconds) << ",\n"
      << "  \"bounds\": {\"min\": [" << jnum(r.mn.x) << ", " << jnum(r.mn.y) << ", " << jnum(r.mn.z)
      << "], \"max\": [" << jnum(r.mx.x) << ", " << jnum(r.mx.y) << ", " << jnum(r.mx.z) << "]},\n"
      << "  \"pipegen\": {\"pipeline_generated_by\": \"PipeGen v0.5.1\", \"engine_built_by\": \"PipeGen v0.5.1\", "
      << "\"hand_completed_stages\": [\"VggtPair\", \"VggtDecode\"]}\n}\n";
    return m.str();
}

void draw(NvDsCustomPostProcessFrameInCpu* in, const cv::Mat& view, const float* depth, const std::string& text) {
    cv::Mat canvas(in->H, in->W, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat left; cv::resize(view, left, cv::Size(in->W / 2, in->H));
    left.copyTo(canvas(cv::Rect(0, 0, in->W / 2, in->H)));
    if (depth) {
        cv::Mat f(S, S, CV_32F, (void*) depth), u8, color;
        double mn, mx; cv::minMaxLoc(f, &mn, &mx);
        double sc = mx > mn ? 255.0 / (mx - mn) : 0;
        f.convertTo(u8, CV_8U, sc, -mn * sc);
        cv::applyColorMap(u8, color, cv::COLORMAP_INFERNO);
        cv::resize(color, color, cv::Size(in->W - in->W / 2, in->H));
        color.copyTo(canvas(cv::Rect(in->W / 2, 0, in->W - in->W / 2, in->H)));
    }
    cv::rectangle(canvas, cv::Rect(0, 0, in->W, 30), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, text, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    write_nv12(canvas, in);
}

}  // namespace

extern "C" bool VggtPair(NvDsCustomPostProcessFrameInCpu* in) {
    if (!in || !in->Y_host || in->W != 2 * S || in->H != S) {
        fprintf(stderr, "[vggt] VggtPair needs a %dx%d surface (set nvstreammux width/height)\n", 2 * S, S);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    init_locked();
    const int f = frame_num(in);
    cv::Mat view = make_view(read_nv12_bgr(in));
    g_views[f] = view;
    Pair pr{f - g_cfg.stride, f};
    if (!g_views.count(pr.first)) pr.first = f;          // warm-up: no frame from 1 s ago yet
    g_pairs[f] = pr;
    while (!g_views.empty() && g_views.begin()->first < f - 2 * g_cfg.stride - 4) g_views.erase(g_views.begin());
    cv::Mat atlas(S, 2 * S, CV_8UC3);
    g_views[pr.first].copyTo(atlas(cv::Rect(0, 0, S, S)));
    view.copyTo(atlas(cv::Rect(S, 0, S, S)));
    write_nv12(atlas, in);
    return true;
}

extern "C" bool VggtDecode(NvDsCustomPostProcessFrameInCpu* in) {
    if (!in || !in->Y_host) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    init_locked();
    auto now = std::chrono::steady_clock::now();
    double frame_seconds = std::chrono::duration<double>(now - g_last).count(); g_last = now;
    const int f = frame_num(in);
    auto pit = g_pairs.find(f);
    if (pit == g_pairs.end() || !g_views.count(pit->second.first) || !g_views.count(f)) {
        fprintf(stderr, "[vggt] frame %d: no atlas record\n", f); return true;
    }
    Pair pr = pit->second;
    cv::Mat views[2] = {g_views[pr.first], g_views[pr.second]};
    Tensor pose, depth, conf;
    if (!get_tensor(in, "pose_enc", pose) || !get_tensor(in, "depth", depth) || !get_tensor(in, "depth_conf", conf) ||
        pose.v.size() != 18 || depth.v.size() != 2 * (size_t) S * S || conf.v.size() != depth.v.size()) {
        fprintf(stderr, "[vggt] frame %d: missing/mis-shaped VGGT outputs\n", f);
        draw(in, views[1], nullptr, "VGGT outputs missing");
        return true;
    }
    for (float x : pose.v) if (!std::isfinite(x)) { fprintf(stderr, "[vggt] frame %d: non-finite pose\n", f); return true; }
    const float* depth_cur = &depth.v[(size_t) S * S];
    char text[200];
    if (pr.first == pr.second) {
        snprintf(text, sizeof text, "frame %d | VGGT FP16 TensorRT | warm-up: waiting for frame t-%.1fs", f, g_cfg.stride / g_cfg.fps);
        fprintf(stderr, "[vggt] frame %d: warm-up (single view), %.3fs\n", f, frame_seconds);
        draw(in, views[1], depth_cur, text);
        return true;
    }
    Camera cams[2]; Cloud cloud; Result res;
    if (!reconstruct(pose, depth, conf, views, cams, cloud, res)) { draw(in, views[1], depth_cur, "reconstruction failed"); return true; }
    char name[64]; snprintf(name, sizeof name, "pair_%04d_%04d", pr.first, pr.second);
    fs::path dir = fs::path(g_cfg.out) / "pairs", ply = dir / (std::string(name) + ".ply");
    if (!write_ply(ply, cloud) || !verify_ply(ply, cloud.p.size())) { fprintf(stderr, "[vggt] PLY write/verify failed: %s\n", ply.c_str()); return true; }
    std::string meta = metadata_json(pr, cams, res, ply.string(), frame_seconds);
    std::ofstream(dir / (std::string(name) + ".json")) << meta;
    fs::path out = g_cfg.out;
    if (!g_wrote_primary) {      // first full pair = the reference's sampling (frames 0 and t+1s)
        fs::copy_file(ply, out / "cloud.ply", fs::copy_options::overwrite_existing);
        std::string primary = meta;
        size_t at = primary.find(ply.string());
        if (at != std::string::npos) primary.replace(at, ply.string().size(), (out / "cloud.ply").string());
        std::ofstream(out / "metadata.json") << primary;
        g_wrote_primary = true;
    }
    g_pairs_written++;
    double t = std::chrono::duration<double>(now - g_t0).count();
    std::ofstream ev(out / "events.jsonl", std::ios::app);
    ev << "{\"schema_version\": 1, \"event\": \"geometry_ready\", \"action\": \"inspect_reconstruction\", "
       << "\"backend\": \"tensorrt-deepstream\", \"frames\": [" << pr.first << ", " << pr.second << "], "
       << "\"point_count\": " << res.points << ", \"scale\": \"arbitrary; uncalibrated\", "
       << "\"ply\": \"" << ply.string() << "\", \"metadata\": \"" << (dir / (std::string(name) + ".json")).string() << "\", "
       << "\"stream_seconds\": " << jnum(t) << ", \"frame_wall_seconds\": " << jnum(frame_seconds) << "}\n";
    fprintf(stderr, "[vggt] frame %d: pair (%d,%d) -> %zu points (conf>=%.4f), %.3fs -> %s\n",
            f, pr.first, pr.second, res.points, res.threshold, frame_seconds, ply.c_str());
    snprintf(text, sizeof text, "frame %d | VGGT FP16 TensorRT | views %d+%d -> %zu pts", f, pr.first, pr.second, res.points);
    draw(in, views[1], depth_cur, text);
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(VggtPair);
CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(VggtDecode);

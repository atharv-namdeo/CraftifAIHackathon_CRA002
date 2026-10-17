# Builds the SPECTRA nvcustomprocess library plus PipeGen's generated meta_json writer.
# Flags are PipeGen's generated NVCP_* / JSON_* flags (DeepStream 7.1 image).
DS_VER   ?= 7.1
CXX      := g++
DS_INC   := /opt/nvidia/deepstream/deepstream-$(DS_VER)/sources/includes
DS_LIB   := /opt/nvidia/deepstream/deepstream-$(DS_VER)/lib
OUT_DIR  ?= ../build

NVCP_PKGS := glib-2.0 gstreamer-1.0 opencv4
NVCP_CFLAGS := -fPIC -O2 -std=c++17 -Wall -I $(DS_INC) -I /usr/local/include \
  $(shell pkg-config --cflags $(NVCP_PKGS))
NVCP_LDFLAGS := -shared -Wl,--no-undefined -L $(DS_LIB) -lnvds_meta -Wl,-rpath,$(DS_LIB) \
  $(shell pkg-config --libs $(NVCP_PKGS))

JSON_CFLAGS  := -fPIC -O2 -std=c++14 -Wall -I $(DS_INC) -I /usr/local/include \
  $(shell pkg-config --cflags glib-2.0)
JSON_LDFLAGS := -shared -Wl,--no-undefined -L $(DS_LIB) -lnvds_meta -Wl,-rpath,$(DS_LIB) \
  $(shell pkg-config --libs glib-2.0)

all: $(OUT_DIR)/libvggt_spectra.so $(OUT_DIR)/libmeta_json.so

$(OUT_DIR)/libvggt_spectra.so: vggt_spectra.cpp Makefile
	mkdir -p $(OUT_DIR)
	$(CXX) $(NVCP_CFLAGS) -o $@ vggt_spectra.cpp $(NVCP_LDFLAGS)

$(OUT_DIR)/libmeta_json.so: meta_json.cpp frame_json.h render_meta.h Makefile
	mkdir -p $(OUT_DIR)
	$(CXX) $(JSON_CFLAGS) -o $@ meta_json.cpp $(JSON_LDFLAGS)

clean:
	rm -f $(OUT_DIR)/libvggt_spectra.so $(OUT_DIR)/libmeta_json.so

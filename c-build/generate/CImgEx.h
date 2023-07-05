#ifndef cimg_plugin
#define cimg_plugin "CImgEx.h"
#include <vector>

#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"

#include "stb_image_write.h"

#define cimg_load_plugin(filename) \
    if (!cimg::strcasecmp(ext,"jpg") \
    ||  !cimg::strcasecmp(ext,"jpeg") \
    ||  !cimg::strcasecmp(ext,"jpe") \
    ||  !cimg::strcasecmp(ext,"jfif") \
    ||  !cimg::strcasecmp(ext,"jif") \
    ||  !cimg::strcasecmp(ext,"png")) return load_from_file(filename);

#define cimg_save_plugin(filename) \
    if (!cimg::strcasecmp(ext,"jpg") \
    ||  !cimg::strcasecmp(ext,"jpeg") \
    ||  !cimg::strcasecmp(ext,"jpe") \
    ||  !cimg::strcasecmp(ext,"jfif") \
    ||  !cimg::strcasecmp(ext,"jif") \
    ||  !cimg::strcasecmp(ext,"png")) return save_to_file(filename); \

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
void stbi_write_vector(void* context, void* data, int size);
#else
void stbi_write_vector(void* context, void* data, int size)
{
    auto ptr = reinterpret_cast<unsigned char*>(data);
    auto mem = reinterpret_cast<std::vector<unsigned char>*>(context);
    for (int i = 0; i < size; i++) {
        mem->push_back(*ptr++);
    }
}
#endif

#include "CImg.h"

#else
/**************************************************************************}}}*/
/*** CImg Plugins:                                                          ***/
/**************************************************************************{{{*/
CImg<T>& load_from_file(const char *const filename)
{
  int x, y, n;
  unsigned char* data = stbi_load(filename, &x, &y, &n, 0);
  if (data == NULL) {
    throw CImgIOException(_cimg_instance
                          "load_from_file: %s.",
                          cimg_instance, stbi_failure_reason());
  }

  try { assign(x, y, 1, n); } catch (...) { throw; }

  read_hwc_from(data);

  stbi_image_free(data);

  return *this;
}

void read_hwc_from(const unsigned char* ptrs)
{
  switch (_spectrum) {
  case 1: {
      T *ptr_r = _data;
      cimg_forXY(*this, x, y) {
        *(ptr_r++) = (T)*(ptrs++);
      }
    }
    break;
  case 3: {
      T *ptr_r = _data,
        *ptr_g = _data + 1UL*_width*_height,
        *ptr_b = _data + 2UL*_width*_height;
      cimg_forXY(*this, x, y) {
        *(ptr_r++) = (T)*(ptrs++);
        *(ptr_g++) = (T)*(ptrs++);
        *(ptr_b++) = (T)*(ptrs++);
      }
    }
    break;
  case 4: {
      T *ptr_r = _data,
        *ptr_g = _data + 1UL*_width*_height,
        *ptr_b = _data + 2UL*_width*_height,
        *ptr_a = _data + 3UL*_width*_height;
      cimg_forXY(*this, x, y) {
        *(ptr_r++) = (T)*(ptrs++);
        *(ptr_g++) = (T)*(ptrs++);
        *(ptr_b++) = (T)*(ptrs++);
        *(ptr_a++) = (T)*(ptrs++);
      }
    }
    break;
  }
}

const CImg<T>& save_to_file(const char *const filename) const
{
  if (is_empty()) { return *this; }
  if (_depth > 1) {
    cimg::warn(_cimg_instance
               "save_to_file(): Instance is volumetric, only the first slice will be saved in file '%s'.",
               cimg_instance,
               filename);
  }

  unsigned char *buff = reinterpret_cast<unsigned char*>(malloc(_width*_height*_spectrum));
  if (buff == NULL) {
    throw CImgIOException(_cimg_instance
                           "save_to_file: Failed to allocate memory for work.",
                           cimg_instance);
  }

  write_hwc_to(buff);

  const char *const ext = cimg::split_filename(filename);
  if (cimg::strcasecmp(ext,"png") == 0) {
      stbi_write_png(filename, _width, _height, _spectrum, buff, 0);
  }
  else {
      stbi_write_jpg(filename, _width, _height, _spectrum, buff, 100);
  }
  return *this;
}

void write_hwc_to(unsigned char* ptrd) const
{
  switch (_spectrum) {
  case 1: {
      const T *ptr_g = data(0, 0, 0, 0);
      cimg_forXY(*this, x, y) {
        *(ptrd++) = (unsigned char)*(ptr_g++);
      }
    }
    break;
  case 3: {
      const T *ptr_r = data(0, 0, 0, 0),
               *ptr_g = data(0, 0, 0, 1),
               *ptr_b = data(0, 0, 0, 2);
      cimg_forXY(*this, x, y) {
        *(ptrd++) = (unsigned char)*(ptr_r++);
        *(ptrd++) = (unsigned char)*(ptr_g++);
        *(ptrd++) = (unsigned char)*(ptr_b++);
      }
    }
    break;
  case 4: {
      const T *ptr_r = data(0, 0, 0, 0),
               *ptr_g = data(0, 0, 0, 1),
               *ptr_b = data(0, 0, 0, 2),
               *ptr_a = data(0, 0, 0, 3);
      cimg_forXY(*this, x, y) {
        *(ptrd++) = (unsigned char)*(ptr_r++);
        *(ptrd++) = (unsigned char)*(ptr_g++);
        *(ptrd++) = (unsigned char)*(ptr_b++);
        *(ptrd++) = (unsigned char)*(ptr_a++);
      }
    }
    break;
  }
}
#endif
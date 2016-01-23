#ifndef _pv_alloc_h
#define _pv_alloc_h

#include <stdlib.h>

#define malloc(size) pv_malloc(__FILE__, __LINE__, size)
#define calloc(count, size) pv_calloc(__FILE__, __LINE__, count, size)
#define malloc_message(size, fmt, ...) pv_malloc_message(__FILE__, __LINE__, size, fmt, ##__VA_ARGS__)
#define calloc_message(size, fmt, ...) pv_calloc_message(__FILE__, __LINE__, count * size, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void *pv_malloc(const char *file, int line, size_t size);
void *pv_malloc_message(const char *file, int line, size_t size, const char *fmt, ...); 

void *pv_calloc(const char *file, int line, size_t count, size_t size);
void *pv_calloc_message(const char *file, int line, size_t count, size_t size, const char *fmt, ...);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
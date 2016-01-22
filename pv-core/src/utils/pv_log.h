#ifndef _pv_log_h
#define _pv_log_h

#ifdef DEBUG
#define LOG_DEBUG_OUTPUT 1
#endif

#ifdef LOG_DEBUG_OUTPUT
#define log_debug(fmt, ...) pv_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define log_debug(fmt, ...)
#endif

#define log_error(fmt, ...) pv_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void pv_log_error(const char *file, int line, const char *fmt, ...);
void pv_log_debug(const char *file, int line, const char *fmt, ...);

// Non-varargs versions
void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
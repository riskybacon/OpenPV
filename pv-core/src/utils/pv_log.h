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

#define pv_assert(c, fmt, ...) if (!(c)) { pv_assert_failed(__FILE__, __LINE__, #c, fmt, ##__VA_ARGS__); }

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void pv_log_error(const char *file, int line, const char *fmt, ...);
void pv_log_debug(const char *file, int line, const char *fmt, ...);

// Non-varargs versions
void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

void pv_assert_failed(const char *file, int line, const char *condition, const char *fmt, ...);

#if 0
#include <execinfo.h>
#include <cxxabi.h>

   /** Print a demangled stack backtrace of the caller function to FILE* out. */
   static inline void print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63)
   {
      fprintf(out, "stack trace:\n");

      // storage array for stack trace address data
      void* addrlist[max_frames+1];

      // retrieve current stack addresses
      int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

      if (addrlen == 0) {
         fprintf(out, "  <empty, possibly corrupt>\n");
         return;
      }

      // resolve addresses into strings containing "filename(function+address)",
      // this array must be free()-ed
      char** symbollist = backtrace_symbols(addrlist, addrlen);

      // allocate string which will be filled with the demangled function name
      size_t funcnamesize = 256;
      char* funcname = (char*)malloc(funcnamesize);

      // iterate over the returned symbol lines. skip the first, it is the
      // address of this function.
      for (int i = 1; i < addrlen; i++)
      {
         char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

         // find parentheses and +address offset surrounding the mangled name:
         // ./module(function+0x15c) [0x8048a6d]
         for (char *p = symbollist[i]; *p; ++p)
         {
            if (*p == '(')
               begin_name = p;
            else if (*p == '+')
               begin_offset = p;
            else if (*p == ')' && begin_offset) {
               end_offset = p;
               break;
            }
         }

         if (begin_name && begin_offset && end_offset
             && begin_name < begin_offset)
         {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():

            int status;
            char* ret = abi::__cxa_demangle(begin_name,
                                            funcname, &funcnamesize, &status);
            if (status == 0) {
               funcname = ret; // use possibly realloc()-ed string
               fprintf(out, "  %s : %s+%s\n",
                       symbollist[i], funcname, begin_offset);
            }
            else {
               // demangling failed. Output function name as a C function with
               // no arguments.
               fprintf(out, "  %s : %s()+%s\n",
                       symbollist[i], begin_name, begin_offset);
            }
         }
         else
         {
            // couldn't parse the line? print the whole line.
            fprintf(out, "  %s\n", symbollist[i]);
         }
      }
      
      free(funcname);
      free(symbollist);
   }
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
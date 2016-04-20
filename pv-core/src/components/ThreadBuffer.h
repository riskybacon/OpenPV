#ifndef THREAD_BUFFER_H_
#define THREAD_BUFFER_H_

#ifdef PV_USE_OPENMP_THREADS

#include "components/Boilerplate.h"

namespace PV {

void accumVecs(float *dest, float * src, size_t count) {
   for (int idx = 0; idx < count; idx++) {
      dest[idx] += src[idx];
   }
   return dest;
}

#pragma omp declare reduction(accumVecs: float : omp_out=accumVecs(&omp_out, &omp_in))
   
/**
 * Thread reduction buffer
 */
template <typename T = float>
class ThreadBuffer {
   // Accumulation buffer for each thread
   std::vector< std::vector<T> > mBuffer;
   // Number of threads (size of first dimension)
   int mNumThreads = 0;
   // Number of elements for each thread (size of second dimension)
   int mNumElements = 0;
   // Initial value
   T mInitialValue;

   // Copy and assign are not allowed
   PV_DISALLOW_COPY_AND_ASSIGN(ThreadBuffer);

public:
   /**
    * Default constructor
    *
    * Creates an empty buffer
    */
   ThreadBuffer(T initialValue = 0)
   : mInitialValue(initialValue)
   {
   }

   /**
    * Constructor
    *
    * @param numThreads   number of threads
    * @param numElements  number of elements for each thread
    * @param initialValue initial value for all elements
    */
   ThreadBuffer(int numThreads, int numElements, T initialValue)
   : mNumThreads(numThreads)
   , mNumElements(numElements)
   , mInitialValue(initialValue)
   {
      resize(mNumThreads, mNumElements);
      reset();
   }

   /**
    * @return a reference to the element at position n in the container
    */
   std::vector<T>& operator[](size_t n) {
      return mBuffer[n];
   }

   /**
    * @return a const reference to the element at position n in the container
    */
   const std::vector<T>& operator[](size_t n) const {
      return mBuffer[n];
   }

   /**
    * @return the number of thread buffers
    */
   size_t size() const {
      return mBuffer.size();
   }

   /**
    * Reset the container to the initial value
    */
   void reset() {
      //#pragma omp parallel for
      for (int i = 0; i < mNumThreads * mNumElements; i++) {
         int ti = i / mNumElements;
         int ni = i % mNumElements;
         mBuffer[ti][ni] = mInitialValue;
      }
   }

   /**
    * Resize the container. Values are not set to the initial value
    *
    * @param numThreads   number of threads
    * @param numElements  number of elements for each thread
    */
   void resize(int numThreads, int numElements) {
      mNumThreads = numThreads;
      mNumElements = numElements;

      mBuffer.resize(mNumThreads);

      for (auto& buf : mBuffer) {
         buf.resize(mNumElements);
      }
   }

   /**
    * Reduce the the buffers into a destination
    */
   void reduce(T * dest) {
      // Memory access patterns may be inefficient here. However, there are
      // no collisions when updating the post synaptic perspective, so no
      // atomics or locking required.
//#pragma omp parallel for
//      for(int ni = 0; ni < mNumElements; ni++){
//         for(int ti = 0; ti < mNumThreads; ti++){
//            dest[ni] += mBuffer[ti][ni];
//         }
//      }

      for(int ti = 0; ti < mNumThreads; ti++) {
#pragma omp parallel for schedule(static,7)
         for(int ni = 0; ni < mNumElements; ni++) {
            dest[ni] += mBuffer[ti][ni];
         }
      }

#pragma omp parallel for reduction(accumVecs: dest)
      for ( n=0 ; n<10 ; ++n )
      {
         for (m=0; m<=n; ++m){
            S.v[n] += A[m];
         }
      }

   }
};
} // namespace PV

#endif // PV_USE_OPENMP_THREADS

#endif // THREAD_BUFFER_H_

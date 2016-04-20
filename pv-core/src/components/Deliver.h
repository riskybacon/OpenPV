#ifndef DELIVER_H_
#define DELIVER_H_

#include <type_traits>
#include "components/Deliver.h"
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"

#define PV_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName& source) = delete; \
  TypeName& operator=(const TypeName& source) = delete

namespace PV {

/**
 * Base delivery class. Delivers the source synaptic perspective to the target synaptic
 * layer. The base class does very little and should not be used directly.
 * Instead, create a subclasss, preferably a subclass that is templatized for
 * data types, gSyn accumulation types, patch geometries, accumlation functors, etc.
 */
class Deliver {
public:
   PV_DISALLOW_COPY_AND_ASSIGN(Deliver);
   virtual void operator()(PVLayerCube const * activity, int arborID, int* numActive = NULL, int** activeList = NULL) = 0;
protected:
   Deliver() {}
};

} // Namespace PV

#endif // _DELIVER_H


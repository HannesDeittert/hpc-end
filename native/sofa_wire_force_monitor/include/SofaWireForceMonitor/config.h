#pragma once

#include <sofa/config.h>
#include <sofa/config/sharedlibrary_defines.h>

#ifdef SOFA_BUILD_SOFA_WIRE_FORCE_MONITOR
#define SOFA_TARGET SofaWireForceMonitor
#define SOFA_WIRE_FORCE_MONITOR_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_WIRE_FORCE_MONITOR_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

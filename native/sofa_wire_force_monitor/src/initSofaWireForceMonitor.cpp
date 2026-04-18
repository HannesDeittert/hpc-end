#include <SofaWireForceMonitor/WireWallContactExport.h>
#include <SofaWireForceMonitor/WireWallForceMonitor.h>
#include <SofaWireForceMonitor/config.h>

#include <mutex>

namespace
{
void initSofaWireForceMonitor()
{
    // registration happens through static initialization in WireWallForceMonitor.cpp
}
}

extern "C"
{
SOFA_WIRE_FORCE_MONITOR_API void initExternalModule()
{
    static std::once_flag once;
    std::call_once(once, []() { initSofaWireForceMonitor(); });
}

SOFA_WIRE_FORCE_MONITOR_API const char* getModuleName()
{
    return "SofaWireForceMonitor";
}

SOFA_WIRE_FORCE_MONITOR_API const char* getModuleVersion()
{
    return "0.1.0";
}

SOFA_WIRE_FORCE_MONITOR_API const char* getModuleLicense()
{
    return "LGPL";
}

SOFA_WIRE_FORCE_MONITOR_API const char* getModuleDescription()
{
    return "Wall-force and explicit wall-contact export monitors for full-wire telemetry.";
}

SOFA_WIRE_FORCE_MONITOR_API const char* getModuleComponentList()
{
    return "WireWallForceMonitor, WireWallContactExport";
}
}

#pragma once

#include <SofaWireForceMonitor/config.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>

#include <string>

namespace sofa::component::monitor
{

class SOFA_WIRE_FORCE_MONITOR_API WireWallForceMonitor final
    : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(WireWallForceMonitor, sofa::core::objectmodel::BaseObject);

    using Vec3 = sofa::type::Vec3d;
    using Vec3List = sofa::type::vector<Vec3>;
    using IndexList = sofa::type::vector<unsigned int>;
    using MechanicalState = sofa::core::behavior::BaseMechanicalState;
    using MeshTopology = sofa::core::topology::BaseMeshTopology;

    WireWallForceMonitor();
    ~WireWallForceMonitor() override = default;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

private:
    void updateTelemetry();
    bool readStatePositions3(const MechanicalState* state, Vec3List& out) const;
    bool readStateForces3(
        const MechanicalState* state,
        sofa::core::ConstVecDerivId forceId,
        Vec3List& out) const;
    bool updateWallTriangleCentroids();
    unsigned int findNearestWallTriangle(const Vec3& p) const;
    void setUnavailable(const std::string& reason);

    sofa::core::objectmodel::SingleLink<
        WireWallForceMonitor,
        MechanicalState,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_collisionMechanicalObject;
    sofa::core::objectmodel::SingleLink<
        WireWallForceMonitor,
        MechanicalState,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_wireMechanicalObject;
    sofa::core::objectmodel::SingleLink<
        WireWallForceMonitor,
        MechanicalState,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_vesselMechanicalObject;
    sofa::core::objectmodel::SingleLink<
        WireWallForceMonitor,
        MeshTopology,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_vesselTopology;

    sofa::Data<double> d_contactEpsilon;

    sofa::Data<Vec3List> d_contactForceVectors;
    sofa::Data<IndexList> d_contactSegmentIndices;
    sofa::Data<Vec3List> d_segmentForceVectors;
    sofa::Data<Vec3> d_totalForceVector;
    sofa::Data<double> d_totalForceNorm;
    sofa::Data<unsigned int> d_contactCount;
    sofa::Data<unsigned int> d_wallSegmentCount;
    sofa::Data<IndexList> d_wallActiveSegmentIds;
    sofa::Data<Vec3List> d_wallActiveSegmentForceVectors;
    sofa::Data<Vec3> d_wallTotalForceVector;
    sofa::Data<bool> d_available;
    sofa::Data<std::string> d_source;
    sofa::Data<std::string> d_status;

    Vec3List m_wallTriangleCentroids;
    sofa::Size m_cachedWallTriangleCount {0};
};

} // namespace sofa::component::monitor

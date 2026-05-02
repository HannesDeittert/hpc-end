#pragma once

#include <SofaWireForceMonitor/config.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <string>
#include <unordered_map>

namespace sofa::simulation
{
class Node;
} // namespace sofa::simulation

namespace sofa::component::monitor
{

class SOFA_WIRE_FORCE_MONITOR_API WireWallContactExport final
    : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(WireWallContactExport, sofa::core::objectmodel::BaseObject);

    using Vec3 = sofa::type::Vec3d;
    using Vec3List = sofa::type::vector<Vec3>;
    using UIntList = sofa::type::vector<unsigned int>;
    using IntList = sofa::type::vector<int>;
    using StringList = sofa::type::vector<std::string>;
    using MechanicalState = sofa::core::behavior::BaseMechanicalState;
    using MeshTopology = sofa::core::topology::BaseMeshTopology;

    WireWallContactExport();
    ~WireWallContactExport() override = default;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

private:
    struct ContactRecord
    {
        Vec3 wallPoint {0.0, 0.0, 0.0};
        std::string sourceNodeTag;
        std::string contactKind;
        unsigned int wallSide {0};
        unsigned int contactLocalIndex {0};
        int wallTriangleId {-1};
        int constraintRowIndex {-1};
        int collisionDofIndex {-1};
        bool triangleIdValid {false};
        bool constraintRowValid {false};
        bool collisionDofValid {false};
        bool inRange {false};
        bool mappingComplete {false};
    };

    struct MapperIndices
    {
        IntList map1d;
        IntList map2d;
        IntList map3d;
        bool parsed {false};
        std::string error;
    };

    static bool readStatePositions3(const MechanicalState* state, Vec3List& out);
    static MapperIndices readMapperIndicesFromMappingObject(
        sofa::core::objectmodel::BaseObject* mappingObj);
    static bool parseMapperSerialization(
        const std::string& serialized,
        MapperIndices& out);
    static std::unordered_map<int, sofa::type::vector<int>> parseConstraintRowsByDof(
        const std::string& serialized);
    static std::unordered_map<int, sofa::type::vector<int>> parseConstraintRowsByRow(
        const std::string& serialized);
    bool collectRecordsFromNode(
        sofa::simulation::Node* node,
        const MechanicalState* vesselMechanicalState,
        const std::string& nodeTag,
        unsigned int wallSide,
        const std::string& wallPrimitive,
        int wallTriangleCount,
        const IntList& vertexToTriangle,
        const IntList& edgeToTriangle,
        sofa::type::vector<ContactRecord>& out) const;
    void updateTelemetry();
    void setUnavailable(const std::string& reason);

    sofa::core::objectmodel::SingleLink<
        WireWallContactExport,
        sofa::simulation::Node,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_vesselNode;
    sofa::core::objectmodel::SingleLink<
        WireWallContactExport,
        MeshTopology,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_vesselTopology;
    sofa::core::objectmodel::SingleLink<
        WireWallContactExport,
        MechanicalState,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK>
        l_collisionMechanicalObject;
    sofa::Data<double> d_contactEpsilon;

    sofa::Data<UIntList> d_contactLocalIndices;
    sofa::Data<IntList> d_wallTriangleIds;
    sofa::Data<Vec3List> d_wallPoints;
    sofa::Data<StringList> d_sourceNodeTags;
    sofa::Data<UIntList> d_modelSides;
    sofa::Data<IntList> d_constraintRowIndices;
    sofa::Data<UIntList> d_constraintRowValidFlags;
    sofa::Data<IntList> d_collisionDofIndices;
    sofa::Data<UIntList> d_collisionDofValidFlags;
    sofa::Data<StringList> d_contactKinds;

    sofa::Data<UIntList> d_triangleIdValidFlags;
    sofa::Data<UIntList> d_inRangeFlags;
    sofa::Data<UIntList> d_mappingCompleteFlags;
    sofa::Data<UIntList> d_orderingStableFlags;

    sofa::Data<unsigned int> d_contactCount;
    sofa::Data<double> d_explicitCoverage;
    sofa::Data<bool> d_orderingStable;
    sofa::Data<bool> d_available;
    sofa::Data<std::string> d_source;
    sofa::Data<std::string> d_status;
};

} // namespace sofa::component::monitor

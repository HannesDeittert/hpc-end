#include <SofaWireForceMonitor/WireWallContactExport.h>

#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/helper/accessor/WriteAccessor.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sofa::component::monitor
{

namespace
{
int WireWallContactExportClass = sofa::core::RegisterObject(
                                     "Native explicit contact->wall-triangle exporter "
                                     "for validated wall-force telemetry.")
                                     .add<WireWallContactExport>();

inline long long quantizeCoord(const double x)
{
    // Deterministic sort quantization for stable contact record ordering.
    return static_cast<long long>(std::llround(x * 1.0e6));
}

inline unsigned int asFlag(const bool v)
{
    return v ? 1U : 0U;
}

inline std::string classifyCollisionModelToken(const std::string& token)
{
    if (token.find("TriangleCollisionModel") != std::string::npos)
    {
        return "triangle";
    }
    if (token.find("LineCollisionModel") != std::string::npos)
    {
        return "line";
    }
    if (token.find("PointCollisionModel") != std::string::npos)
    {
        return "point";
    }
    return "unknown";
}

inline std::uint64_t edgeKey(const int a, const int b)
{
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

inline bool parseContactNodeSignature(
    const std::string& nodeName,
    unsigned int& wallSideOut,
    std::string& contactKindOut,
    std::string& wallPrimitiveOut)
{
    const auto sep = nodeName.find('-');
    if (sep == std::string::npos)
    {
        return false;
    }

    const std::string lhs = nodeName.substr(0, sep);
    const std::string rhs = nodeName.substr(sep + 1);
    const std::string lhsKind = classifyCollisionModelToken(lhs);
    const std::string rhsKind = classifyCollisionModelToken(rhs);
    if (lhsKind == "unknown" && rhsKind == "unknown")
    {
        return false;
    }

    wallSideOut = 0U;
    wallPrimitiveOut = lhsKind;
    if (lhsKind == "unknown" && rhsKind != "unknown")
    {
        wallSideOut = 1U;
        wallPrimitiveOut = rhsKind;
    }
    if (wallPrimitiveOut == "unknown")
    {
        return false;
    }

    if (lhsKind == "line" || rhsKind == "line")
    {
        contactKindOut = "line";
    }
    else if (lhsKind == "point" || rhsKind == "point")
    {
        contactKindOut = "point";
    }
    else
    {
        contactKindOut = "unknown";
    }
    return true;
}

inline int resolveWallTriangleId(
    const int mappedWallElementId,
    const std::string& wallPrimitive,
    const sofa::type::vector<int>& vertexToTriangle,
    const sofa::type::vector<int>& edgeToTriangle)
{
    if (mappedWallElementId < 0)
    {
        return -1;
    }
    if (wallPrimitive == "triangle")
    {
        return mappedWallElementId;
    }
    if (wallPrimitive == "line")
    {
        if (mappedWallElementId >= static_cast<int>(edgeToTriangle.size()))
        {
            return -1;
        }
        return edgeToTriangle[static_cast<sofa::Size>(mappedWallElementId)];
    }
    if (wallPrimitive == "point")
    {
        if (mappedWallElementId >= static_cast<int>(vertexToTriangle.size()))
        {
            return -1;
        }
        return vertexToTriangle[static_cast<sofa::Size>(mappedWallElementId)];
    }
    return -1;
}
} // namespace

WireWallContactExport::WireWallContactExport()
    : l_vesselNode(initLink("vesselNode", "Link to the vessel collision subtree node (typically @vesselTree)."))
    , l_vesselTopology(initLink("vesselTopology", "Link to vessel wall mesh topology (typically @vesselTree/MeshTopology)."))
    , l_collisionMechanicalObject(initLink("collisionMechanicalObject", "Link to collision MechanicalObject (typically @InstrumentCombined/CollisionModel/CollisionDOFs)."))
    , d_contactEpsilon(initData(&d_contactEpsilon, 1e-7, "contactEpsilon", "Threshold for non-zero/active contact checks."))
    , d_contactLocalIndices(initData(&d_contactLocalIndices, "contactLocalIndices", "Per-record local contact index inside source contact node."))
    , d_wallTriangleIds(initData(&d_wallTriangleIds, "wallTriangleIds", "Per-record explicit wall triangle id (-1 if unavailable)."))
    , d_wallPoints(initData(&d_wallPoints, "wallPoints", "Per-record wall-space contact point (Nx3)."))
    , d_sourceNodeTags(initData(&d_sourceNodeTags, "sourceNodeTags", "Per-record source contact-node tag."))
    , d_modelSides(initData(&d_modelSides, "modelSides", "Per-record wall model side flag (0=first model, 1=second model)."))
    , d_constraintRowIndices(initData(&d_constraintRowIndices, "constraintRowIndices", "Per-record explicit constraint row index (-1 if unavailable)."))
    , d_constraintRowValidFlags(initData(&d_constraintRowValidFlags, "constraintRowValidFlags", "Per-record integrity flag: explicit constraint row index valid."))
    , d_collisionDofIndices(initData(&d_collisionDofIndices, "collisionDofIndices", "Per-record collision dof index inferred from row mapping (-1 if unavailable)."))
    , d_collisionDofValidFlags(initData(&d_collisionDofValidFlags, "collisionDofValidFlags", "Per-record integrity flag: collision dof index valid."))
    , d_contactKinds(initData(&d_contactKinds, "contactKinds", "Per-record contact kind: line|point|unknown."))
    , d_triangleIdValidFlags(initData(&d_triangleIdValidFlags, "triangleIdValidFlags", "Per-record integrity flag: explicit triangle id valid."))
    , d_inRangeFlags(initData(&d_inRangeFlags, "inRangeFlags", "Per-record integrity flag: triangle id in [0, wallTriangleCount)."))
    , d_mappingCompleteFlags(initData(&d_mappingCompleteFlags, "mappingCompleteFlags", "Per-record integrity flag: mapper index fully available for record."))
    , d_orderingStableFlags(initData(&d_orderingStableFlags, "orderingStableFlags", "Per-record integrity flag: deterministic ordering key applied."))
    , d_contactCount(initData(&d_contactCount, static_cast<unsigned int>(0), "contactCount", "Number of exported contact records in current step."))
    , d_explicitCoverage(initData(&d_explicitCoverage, 0.0, "explicitCoverage", "Fraction of records with explicit valid wall triangle ids."))
    , d_orderingStable(initData(&d_orderingStable, true, "orderingStable", "True if deterministic ordering applied to current record set."))
    , d_available(initData(&d_available, false, "available", "True when exporter is initialized and current step was processed."))
    , d_source(initData(&d_source, std::string("native_contact_export"), "source", "Source identifier for this telemetry stream."))
    , d_status(initData(&d_status, std::string("not_initialized"), "status", "Status string for diagnostics."))
{
    this->f_listening.setValue(true);
}

void WireWallContactExport::init()
{
    BaseObject::init();
    if (!l_vesselNode)
    {
        setUnavailable("vesselNode link is not set");
        return;
    }
    if (!l_vesselTopology)
    {
        setUnavailable("vesselTopology link is not set");
        return;
    }
    if (!l_collisionMechanicalObject)
    {
        d_status.setValue("collisionMechanicalObject link not set, collision dof mapping unavailable");
    }
    d_source.setValue("native_contact_export");
    if (d_status.getValue() == "not_initialized")
    {
        d_status.setValue("ready");
    }
}

void WireWallContactExport::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        return;
    }
    updateTelemetry();
}

bool WireWallContactExport::readStatePositions3(const MechanicalState* state, Vec3List& out)
{
    out.clear();
    if (state == nullptr)
    {
        return false;
    }
    const auto n = state->getSize();
    const auto dim = state->getCoordDimension();
    if (n == 0 || dim < 3)
    {
        return false;
    }
    const auto scalarCount = static_cast<sofa::Size>(n) * static_cast<sofa::Size>(dim);
    std::vector<::SReal> raw(scalarCount, static_cast<::SReal>(0.0));
    try
    {
        state->copyToBuffer(
            raw.data(),
            sofa::core::ConstVecCoordId::position(),
            static_cast<unsigned int>(scalarCount));
    }
    catch (...)
    {
        return false;
    }

    out.assign(n, Vec3(0.0, 0.0, 0.0));
    for (sofa::Size i = 0; i < n; ++i)
    {
        const auto base = i * static_cast<sofa::Size>(dim);
        out[i] = Vec3(
            static_cast<double>(raw[base]),
            static_cast<double>(raw[base + 1]),
            static_cast<double>(raw[base + 2]));
    }
    return true;
}

bool WireWallContactExport::parseMapperSerialization(
    const std::string& serialized,
    MapperIndices& out)
{
    out = MapperIndices {};
    std::istringstream iss(serialized);
    std::size_t n1 = 0;
    if (!(iss >> n1))
    {
        out.error = "failed to parse map1d size";
        return false;
    }
    out.map1d.reserve(n1);
    for (std::size_t i = 0; i < n1; ++i)
    {
        int inIdx = -1;
        double b0 = 0.0;
        if (!(iss >> inIdx >> b0))
        {
            out.error = "failed to parse map1d entry";
            return false;
        }
        out.map1d.push_back(inIdx);
    }

    std::size_t n2 = 0;
    if (!(iss >> n2))
    {
        out.error = "failed to parse map2d size";
        return false;
    }
    out.map2d.reserve(n2);
    for (std::size_t i = 0; i < n2; ++i)
    {
        int inIdx = -1;
        double b0 = 0.0;
        double b1 = 0.0;
        if (!(iss >> inIdx >> b0 >> b1))
        {
            out.error = "failed to parse map2d entry";
            return false;
        }
        out.map2d.push_back(inIdx);
    }

    std::size_t n3 = 0;
    if (!(iss >> n3))
    {
        out.error = "failed to parse map3d size";
        return false;
    }
    out.map3d.reserve(n3);
    for (std::size_t i = 0; i < n3; ++i)
    {
        int inIdx = -1;
        double b0 = 0.0;
        double b1 = 0.0;
        double b2 = 0.0;
        if (!(iss >> inIdx >> b0 >> b1 >> b2))
        {
            out.error = "failed to parse map3d entry";
            return false;
        }
        out.map3d.push_back(inIdx);
    }
    out.parsed = true;
    out.error.clear();
    return true;
}

std::unordered_map<int, sofa::type::vector<int>>
WireWallContactExport::parseConstraintRowsByDof(const std::string& serialized)
{
    std::unordered_map<int, sofa::type::vector<int>> out;
    std::istringstream input(serialized);
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream ls(line);
        sofa::type::vector<std::string> toks;
        std::string tok;
        while (ls >> tok)
        {
            toks.push_back(tok);
        }
        if (toks.size() < 4)
        {
            continue;
        }
        int rowIdx = -1;
        int nBlocks = 0;
        try
        {
            rowIdx = static_cast<int>(std::stod(toks[0]));
            nBlocks = static_cast<int>(std::stod(toks[1]));
        }
        catch (...)
        {
            continue;
        }
        if (rowIdx < 0 || nBlocks <= 0)
        {
            continue;
        }
        const std::size_t payloadCount = toks.size() - 2;
        if (payloadCount == 0 || payloadCount % static_cast<std::size_t>(nBlocks) != 0)
        {
            continue;
        }
        const std::size_t blockWidth = payloadCount / static_cast<std::size_t>(nBlocks);
        if (blockWidth < 1)
        {
            continue;
        }
        for (int bi = 0; bi < nBlocks; ++bi)
        {
            const std::size_t base = 2 + static_cast<std::size_t>(bi) * blockWidth;
            int dofIdx = -1;
            try
            {
                dofIdx = static_cast<int>(std::stod(toks[base]));
            }
            catch (...)
            {
                continue;
            }
            if (dofIdx < 0)
            {
                continue;
            }
            out[dofIdx].push_back(rowIdx);
        }
    }
    return out;
}

std::unordered_map<int, sofa::type::vector<int>>
WireWallContactExport::parseConstraintRowsByRow(const std::string& serialized)
{
    std::unordered_map<int, sofa::type::vector<int>> out;
    const auto byDof = parseConstraintRowsByDof(serialized);
    for (const auto& kv : byDof)
    {
        const int dof = kv.first;
        for (const int row : kv.second)
        {
            out[row].push_back(dof);
        }
    }
    return out;
}

WireWallContactExport::MapperIndices
WireWallContactExport::readMapperIndicesFromMappingObject(
    sofa::core::objectmodel::BaseObject* mappingObj)
{
    MapperIndices out;
    if (mappingObj == nullptr)
    {
        out.error = "mapping object missing";
        return out;
    }
    auto* mapperLink = mappingObj->findLink("mapper");
    if (mapperLink == nullptr)
    {
        out.error = "mapping.mapper link missing";
        return out;
    }
    auto* linkedBase = mapperLink->getLinkedBase();
    if (linkedBase == nullptr)
    {
        out.error = "mapping.mapper target missing";
        return out;
    }

    using MapperType = sofa::component::mapping::linear::BarycentricMapperMeshTopology<
        sofa::defaulttype::Vec3Types,
        sofa::defaulttype::Vec3Types>;
    auto* mapper = dynamic_cast<MapperType*>(linkedBase);
    if (mapper == nullptr)
    {
        out.error = std::string("unsupported mapper class: ") + linkedBase->getClassName();
        return out;
    }

    std::ostringstream oss;
    oss << *mapper;
    const auto serialized = oss.str();
    if (!parseMapperSerialization(serialized, out))
    {
        if (out.error.empty())
        {
            out.error = "failed to parse mapper serialization";
        }
        return out;
    }
    return out;
}

bool WireWallContactExport::collectRecordsFromNode(
    sofa::simulation::Node* node,
    const std::string& nodeTag,
    const unsigned int wallSide,
    const std::string& wallPrimitive,
    const int wallTriangleCount,
    const IntList& vertexToTriangle,
    const IntList& edgeToTriangle,
    sofa::type::vector<ContactRecord>& out) const
{
    if (node == nullptr)
    {
        return false;
    }

    MechanicalState* contactState = nullptr;
    sofa::core::objectmodel::BaseObject* mappingObj = nullptr;
    for (std::size_t i = 0; i < node->object.size(); ++i)
    {
        auto objPtr = node->object[i];
        auto* baseObj = objPtr.get();
        if (baseObj == nullptr)
        {
            continue;
        }
        if (contactState == nullptr)
        {
            contactState = dynamic_cast<MechanicalState*>(baseObj);
        }
        if (mappingObj == nullptr)
        {
            const std::string className = baseObj->getClassName();
            if (className == "BarycentricMapping")
            {
                mappingObj = baseObj;
            }
        }
    }

    Vec3List positions;
    if (!readStatePositions3(contactState, positions) || positions.empty())
    {
        return false;
    }

    std::unordered_map<int, sofa::type::vector<int>> dofToRows;
    if (contactState != nullptr)
    {
        auto* cData = contactState->findData("constraint");
        if (cData != nullptr)
        {
            dofToRows = parseConstraintRowsByDof(cData->getValueString());
        }
    }

    const auto mapper = readMapperIndicesFromMappingObject(mappingObj);
    const IntList* indexMap = nullptr;
    if (!mapper.map2d.empty())
    {
        indexMap = &mapper.map2d;
    }
    else if (!mapper.map1d.empty())
    {
        indexMap = &mapper.map1d;
    }
    else if (!mapper.map3d.empty())
    {
        indexMap = &mapper.map3d;
    }

    const bool primitiveMapAvailable = wallPrimitive == "triangle"
        || (wallPrimitive == "line" && !edgeToTriangle.empty())
        || (wallPrimitive == "point" && !vertexToTriangle.empty());
    const bool mappingComplete = mapper.parsed && indexMap != nullptr
        && indexMap->size() >= positions.size()
        && primitiveMapAvailable;
    const std::string contactKind =
        (nodeTag.find("LineCollisionModel") != std::string::npos)
            ? "line"
            : ((nodeTag.find("PointCollisionModel") != std::string::npos) ? "point" : "unknown");
    for (sofa::Size i = 0; i < positions.size(); ++i)
    {
        ContactRecord baseRec;
        baseRec.wallPoint = positions[i];
        baseRec.sourceNodeTag = nodeTag;
        baseRec.contactKind = contactKind;
        baseRec.wallSide = wallSide;
        baseRec.contactLocalIndex = static_cast<unsigned int>(i);

        int mappedWallElementId = -1;
        if (indexMap != nullptr && i < indexMap->size())
        {
            mappedWallElementId = (*indexMap)[i];
        }
        const int tri = resolveWallTriangleId(
            mappedWallElementId,
            wallPrimitive,
            vertexToTriangle,
            edgeToTriangle);
        baseRec.wallTriangleId = tri;
        baseRec.inRange = tri >= 0 && tri < wallTriangleCount;
        baseRec.triangleIdValid = baseRec.inRange;
        const bool elementInPrimitiveRange = wallPrimitive == "triangle"
            ? mappedWallElementId >= 0
            : (wallPrimitive == "line"
                ? (mappedWallElementId >= 0
                    && mappedWallElementId < static_cast<int>(edgeToTriangle.size()))
                : (wallPrimitive == "point"
                    ? (mappedWallElementId >= 0
                        && mappedWallElementId < static_cast<int>(vertexToTriangle.size()))
                    : false));
        baseRec.mappingComplete = mappingComplete && elementInPrimitiveRange;

        sofa::type::vector<int> uniqueRows;
        auto rowIt = dofToRows.find(static_cast<int>(i));
        if (rowIt != dofToRows.end() && !rowIt->second.empty())
        {
            std::unordered_set<int> seenRows;
            for (const int row : rowIt->second)
            {
                if (row < 0)
                {
                    continue;
                }
                if (seenRows.insert(row).second)
                {
                    uniqueRows.push_back(row);
                }
            }
            std::sort(uniqueRows.begin(), uniqueRows.end());
        }

        if (uniqueRows.empty())
        {
            ContactRecord rec = baseRec;
            rec.constraintRowValid = false;
            rec.constraintRowIndex = -1;
            out.push_back(rec);
            continue;
        }

        // Export one record per explicit constraint row to allow row-domain
        // deterministic matching in validated force mapping.
        for (const int row : uniqueRows)
        {
            ContactRecord rec = baseRec;
            rec.constraintRowValid = true;
            rec.constraintRowIndex = row;
            out.push_back(rec);
        }
    }
    return true;
}

void WireWallContactExport::updateTelemetry()
{
    auto* vesselNode = l_vesselNode.get();
    auto* vesselTopo = l_vesselTopology.get();
    if (vesselNode == nullptr)
    {
        setUnavailable("vesselNode link resolves to null");
        return;
    }
    if (vesselTopo == nullptr)
    {
        setUnavailable("vesselTopology link resolves to null");
        return;
    }
    const auto wallTriangleCount = static_cast<int>(vesselTopo->getNbTriangles());
    if (wallTriangleCount <= 0)
    {
        setUnavailable("vessel topology has no triangles");
        return;
    }

    IntList vertexToTriangle;
    IntList edgeToTriangle;
    {
        const auto& triangles = vesselTopo->getTriangles();
        int maxVertexIndex = -1;
        std::unordered_map<std::uint64_t, int> edgeToTriangleByKey;
        for (sofa::Size triId = 0; triId < triangles.size(); ++triId)
        {
            const auto& tri = triangles[triId];
            const int v0 = static_cast<int>(tri[0]);
            const int v1 = static_cast<int>(tri[1]);
            const int v2 = static_cast<int>(tri[2]);
            maxVertexIndex = std::max(maxVertexIndex, std::max(v0, std::max(v1, v2)));
            edgeToTriangleByKey.emplace(edgeKey(v0, v1), static_cast<int>(triId));
            edgeToTriangleByKey.emplace(edgeKey(v1, v2), static_cast<int>(triId));
            edgeToTriangleByKey.emplace(edgeKey(v2, v0), static_cast<int>(triId));
        }
        if (maxVertexIndex >= 0)
        {
            vertexToTriangle.assign(static_cast<sofa::Size>(maxVertexIndex + 1), -1);
            for (sofa::Size triId = 0; triId < triangles.size(); ++triId)
            {
                const auto& tri = triangles[triId];
                const int v[3] = {
                    static_cast<int>(tri[0]),
                    static_cast<int>(tri[1]),
                    static_cast<int>(tri[2]),
                };
                for (const int vi : v)
                {
                    if (vi < 0 || vi >= static_cast<int>(vertexToTriangle.size()))
                    {
                        continue;
                    }
                    if (vertexToTriangle[static_cast<sofa::Size>(vi)] < 0)
                    {
                        vertexToTriangle[static_cast<sofa::Size>(vi)] = static_cast<int>(triId);
                    }
                }
            }
        }

        const auto& edges = vesselTopo->getEdges();
        edgeToTriangle.assign(edges.size(), -1);
        for (sofa::Size edgeId = 0; edgeId < edges.size(); ++edgeId)
        {
            const auto& edge = edges[edgeId];
            const int e0 = static_cast<int>(edge[0]);
            const int e1 = static_cast<int>(edge[1]);
            const auto it = edgeToTriangleByKey.find(edgeKey(e0, e1));
            if (it != edgeToTriangleByKey.end())
            {
                edgeToTriangle[edgeId] = static_cast<int>(it->second);
            }
        }
    }

    sofa::type::vector<ContactRecord> records;
    records.reserve(64);
    std::unordered_map<int, sofa::type::vector<int>> collisionRowToDofs;

    auto* collisionState = l_collisionMechanicalObject.get();
    if (collisionState != nullptr)
    {
        auto* cData = collisionState->findData("constraint");
        if (cData == nullptr)
        {
            d_status.setValue("warn:collision_constraint_data_missing");
        }
        else
        {
            collisionRowToDofs = parseConstraintRowsByRow(cData->getValueString());
        }
    }

    struct ContactNodeSpec
    {
        sofa::simulation::Node* node {nullptr};
        std::string tag;
        unsigned int wallSide {0U};
        std::string wallPrimitive {"unknown"};
    };
    std::vector<ContactNodeSpec> contactNodes;
    std::function<void(sofa::simulation::Node*, const std::string&)> gatherNodes;
    gatherNodes = [&](sofa::simulation::Node* current, const std::string& prefix) {
        if (current == nullptr)
        {
            return;
        }
        for (const auto& childRef : current->child)
        {
            auto* childNode = dynamic_cast<sofa::simulation::Node*>(childRef.get());
            if (childNode == nullptr)
            {
                continue;
            }
            const std::string childName = childNode->getName();
            const std::string childPath = prefix.empty()
                ? childName
                : (prefix + "/" + childName);
            unsigned int wallSide = 0U;
            std::string kind;
            std::string wallPrimitive;
            if (parseContactNodeSignature(childName, wallSide, kind, wallPrimitive))
            {
                contactNodes.push_back(ContactNodeSpec {
                    childNode,
                    childPath,
                    wallSide,
                    wallPrimitive,
                });
            }
            gatherNodes(childNode, childPath);
        }
    };
    gatherNodes(vesselNode, vesselNode->getName());

    std::sort(
        contactNodes.begin(),
        contactNodes.end(),
        [](const ContactNodeSpec& a, const ContactNodeSpec& b) {
            return a.tag < b.tag;
        });

    for (const auto& spec : contactNodes)
    {
        if (spec.node == nullptr)
        {
            continue;
        }
        collectRecordsFromNode(
            spec.node,
            spec.tag,
            spec.wallSide,
            spec.wallPrimitive,
            wallTriangleCount,
            vertexToTriangle,
            edgeToTriangle,
            records);
    }

    sofa::type::vector<ContactRecord> expandedRecords;
    expandedRecords.reserve(records.size() * 2);
    for (const auto& rec : records)
    {
        if (!rec.constraintRowValid || rec.constraintRowIndex < 0)
        {
            ContactRecord outRec = rec;
            outRec.collisionDofValid = false;
            outRec.collisionDofIndex = -1;
            expandedRecords.push_back(outRec);
            continue;
        }
        auto it = collisionRowToDofs.find(rec.constraintRowIndex);
        if (it == collisionRowToDofs.end() || it->second.empty())
        {
            ContactRecord outRec = rec;
            outRec.collisionDofValid = false;
            outRec.collisionDofIndex = -1;
            expandedRecords.push_back(outRec);
            continue;
        }
        std::unordered_set<int> uniqueDofs;
        for (const int dof : it->second)
        {
            if (dof >= 0)
            {
                uniqueDofs.insert(dof);
            }
        }
        if (uniqueDofs.empty())
        {
            ContactRecord outRec = rec;
            outRec.collisionDofValid = false;
            outRec.collisionDofIndex = -1;
            expandedRecords.push_back(outRec);
            continue;
        }
        sofa::type::vector<int> sortedDofs;
        sortedDofs.reserve(uniqueDofs.size());
        for (const int dof : uniqueDofs)
        {
            sortedDofs.push_back(dof);
        }
        std::sort(sortedDofs.begin(), sortedDofs.end());
        for (const int dof : sortedDofs)
        {
            ContactRecord outRec = rec;
            outRec.collisionDofValid = true;
            outRec.collisionDofIndex = int(dof);
            expandedRecords.push_back(outRec);
        }
    }
    records.swap(expandedRecords);

    std::sort(
        records.begin(),
        records.end(),
        [](const ContactRecord& a, const ContactRecord& b) {
            const auto keyA = std::make_tuple(
                a.sourceNodeTag,
                a.wallSide,
                a.contactLocalIndex,
                a.constraintRowIndex,
                a.collisionDofIndex,
                a.wallTriangleId,
                quantizeCoord(a.wallPoint[0]),
                quantizeCoord(a.wallPoint[1]),
                quantizeCoord(a.wallPoint[2]));
            const auto keyB = std::make_tuple(
                b.sourceNodeTag,
                b.wallSide,
                b.contactLocalIndex,
                b.constraintRowIndex,
                b.collisionDofIndex,
                b.wallTriangleId,
                quantizeCoord(b.wallPoint[0]),
                quantizeCoord(b.wallPoint[1]),
                quantizeCoord(b.wallPoint[2]));
            return keyA < keyB;
        });

    UIntList localIdx;
    IntList triIds;
    Vec3List wallPoints;
    StringList sourceTags;
    StringList contactKinds;
    UIntList modelSides;
    IntList constraintRows;
    UIntList constraintRowValidFlags;
    IntList collisionDofs;
    UIntList collisionDofValidFlags;
    UIntList triValidFlags;
    UIntList inRangeFlags;
    UIntList mappingCompleteFlags;
    UIntList orderingStableFlags;

    localIdx.reserve(records.size());
    triIds.reserve(records.size());
    wallPoints.reserve(records.size());
    sourceTags.reserve(records.size());
    contactKinds.reserve(records.size());
    modelSides.reserve(records.size());
    constraintRows.reserve(records.size());
    constraintRowValidFlags.reserve(records.size());
    collisionDofs.reserve(records.size());
    collisionDofValidFlags.reserve(records.size());
    triValidFlags.reserve(records.size());
    inRangeFlags.reserve(records.size());
    mappingCompleteFlags.reserve(records.size());
    orderingStableFlags.reserve(records.size());

    std::size_t explicitCount = 0;
    for (const auto& rec : records)
    {
        localIdx.push_back(rec.contactLocalIndex);
        triIds.push_back(rec.wallTriangleId);
        wallPoints.push_back(rec.wallPoint);
        sourceTags.push_back(rec.sourceNodeTag);
        contactKinds.push_back(rec.contactKind);
        modelSides.push_back(rec.wallSide);
        constraintRows.push_back(rec.constraintRowIndex);
        constraintRowValidFlags.push_back(asFlag(rec.constraintRowValid));
        collisionDofs.push_back(rec.collisionDofIndex);
        collisionDofValidFlags.push_back(asFlag(rec.collisionDofValid));
        triValidFlags.push_back(asFlag(rec.triangleIdValid));
        inRangeFlags.push_back(asFlag(rec.inRange));
        mappingCompleteFlags.push_back(asFlag(rec.mappingComplete));
        orderingStableFlags.push_back(1U);
        if (rec.triangleIdValid)
        {
            ++explicitCount;
        }
    }

    const double explicitCoverage = records.empty()
        ? 0.0
        : static_cast<double>(explicitCount) / static_cast<double>(records.size());

    sofa::helper::WriteAccessor<sofa::Data<UIntList>> localIdxAcc(d_contactLocalIndices);
    localIdxAcc.wref() = localIdx;
    sofa::helper::WriteAccessor<sofa::Data<IntList>> triIdsAcc(d_wallTriangleIds);
    triIdsAcc.wref() = triIds;
    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> wallPointsAcc(d_wallPoints);
    wallPointsAcc.wref() = wallPoints;
    sofa::helper::WriteAccessor<sofa::Data<StringList>> sourceTagsAcc(d_sourceNodeTags);
    sourceTagsAcc.wref() = sourceTags;
    sofa::helper::WriteAccessor<sofa::Data<StringList>> contactKindsAcc(d_contactKinds);
    contactKindsAcc.wref() = contactKinds;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> modelSidesAcc(d_modelSides);
    modelSidesAcc.wref() = modelSides;
    sofa::helper::WriteAccessor<sofa::Data<IntList>> constraintRowsAcc(d_constraintRowIndices);
    constraintRowsAcc.wref() = constraintRows;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> constraintRowValidAcc(d_constraintRowValidFlags);
    constraintRowValidAcc.wref() = constraintRowValidFlags;
    sofa::helper::WriteAccessor<sofa::Data<IntList>> collisionDofsAcc(d_collisionDofIndices);
    collisionDofsAcc.wref() = collisionDofs;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> collisionDofValidAcc(d_collisionDofValidFlags);
    collisionDofValidAcc.wref() = collisionDofValidFlags;

    sofa::helper::WriteAccessor<sofa::Data<UIntList>> triValidAcc(d_triangleIdValidFlags);
    triValidAcc.wref() = triValidFlags;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> inRangeAcc(d_inRangeFlags);
    inRangeAcc.wref() = inRangeFlags;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> mappingCompleteAcc(d_mappingCompleteFlags);
    mappingCompleteAcc.wref() = mappingCompleteFlags;
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> orderingStableAcc(d_orderingStableFlags);
    orderingStableAcc.wref() = orderingStableFlags;

    d_contactCount.setValue(static_cast<unsigned int>(records.size()));
    d_explicitCoverage.setValue(explicitCoverage);
    d_orderingStable.setValue(true);
    d_available.setValue(true);
    d_source.setValue("native_contact_export");
    if (records.empty())
    {
        d_status.setValue("ok:no_contacts");
    }
    else
    {
        d_status.setValue(
            "ok:records=" + std::to_string(records.size())
            + ":explicit=" + std::to_string(explicitCount));
    }
}

void WireWallContactExport::setUnavailable(const std::string& reason)
{
    d_available.setValue(false);
    d_source.setValue("native_contact_export");
    d_status.setValue(reason);
    d_contactCount.setValue(0U);
    d_explicitCoverage.setValue(0.0);
    d_orderingStable.setValue(false);

    sofa::helper::WriteAccessor<sofa::Data<UIntList>> localIdxAcc(d_contactLocalIndices);
    localIdxAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<IntList>> triIdsAcc(d_wallTriangleIds);
    triIdsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> wallPointsAcc(d_wallPoints);
    wallPointsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<StringList>> sourceTagsAcc(d_sourceNodeTags);
    sourceTagsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<StringList>> contactKindsAcc(d_contactKinds);
    contactKindsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> modelSidesAcc(d_modelSides);
    modelSidesAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<IntList>> constraintRowsAcc(d_constraintRowIndices);
    constraintRowsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> constraintRowValidAcc(d_constraintRowValidFlags);
    constraintRowValidAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<IntList>> collisionDofsAcc(d_collisionDofIndices);
    collisionDofsAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> collisionDofValidAcc(d_collisionDofValidFlags);
    collisionDofValidAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> triValidAcc(d_triangleIdValidFlags);
    triValidAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> inRangeAcc(d_inRangeFlags);
    inRangeAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> mappingCompleteAcc(d_mappingCompleteFlags);
    mappingCompleteAcc.wref().clear();
    sofa::helper::WriteAccessor<sofa::Data<UIntList>> orderingStableAcc(d_orderingStableFlags);
    orderingStableAcc.wref().clear();
}

} // namespace sofa::component::monitor

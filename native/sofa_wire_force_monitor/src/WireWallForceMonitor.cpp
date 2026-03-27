#include <SofaWireForceMonitor/WireWallForceMonitor.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>
#include <sofa/helper/accessor/WriteAccessor.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace sofa::component::monitor
{

namespace
{
int WireWallForceMonitorClass =
    sofa::core::RegisterObject(
        "Passive monitor that extracts wall-contact forces over the full wire.")
        .add<WireWallForceMonitor>();

inline double vecNorm(const WireWallForceMonitor::Vec3& v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
} // namespace

WireWallForceMonitor::WireWallForceMonitor()
    : l_collisionMechanicalObject(initLink(
          "collisionMechanicalObject",
          "Link to collision MechanicalObject (typically InstrumentCombined/CollisionModel/CollisionDOFs)."))
    , l_wireMechanicalObject(initLink(
          "wireMechanicalObject",
          "Link to wire MechanicalObject (typically InstrumentCombined/DOFs)."))
    , l_vesselMechanicalObject(initLink(
          "vesselMechanicalObject",
          "Link to vessel wall MechanicalObject (typically vesselTree/dofs)."))
    , l_vesselTopology(initLink(
          "vesselTopology",
          "Link to vessel wall topology (typically vesselTree/MeshTopology)."))
    , d_contactEpsilon(initData(&d_contactEpsilon, 1e-7, "contactEpsilon", "Threshold for non-zero contact force detection."))
    , d_contactForceVectors(initData(&d_contactForceVectors, "contactForceVectors", "Per-contact force vectors (Nx3)."))
    , d_contactSegmentIndices(initData(&d_contactSegmentIndices, "contactSegmentIndices", "Per-contact segment index (N)."))
    , d_segmentForceVectors(initData(&d_segmentForceVectors, "segmentForceVectors", "Per-wall-segment force vectors over vessel triangles (Ntriangles x 3)."))
    , d_totalForceVector(initData(&d_totalForceVector, "totalForceVector", "Total wall force vector (sum over active vessel wall segments)."))
    , d_totalForceNorm(initData(&d_totalForceNorm, 0.0, "totalForceNorm", "Norm of total wall force vector."))
    , d_contactCount(initData(&d_contactCount, static_cast<unsigned int>(0), "contactCount", "Count of active wall segments above epsilon."))
    , d_wallSegmentCount(initData(&d_wallSegmentCount, static_cast<unsigned int>(0), "wallSegmentCount", "Total number of vessel wall triangle segments."))
    , d_wallActiveSegmentIds(initData(&d_wallActiveSegmentIds, "wallActiveSegmentIds", "Indices of active wall triangle segments."))
    , d_wallActiveSegmentForceVectors(initData(&d_wallActiveSegmentForceVectors, "wallActiveSegmentForceVectors", "Force vectors for active wall triangle segments."))
    , d_wallTotalForceVector(initData(&d_wallTotalForceVector, "wallTotalForceVector", "Total wall force vector (sum over active wall segments)."))
    , d_available(initData(&d_available, false, "available", "True when force telemetry is available for current step."))
    , d_source(initData(&d_source, std::string("passive_monitor"), "source", "Source identifier for this telemetry stream."))
    , d_status(initData(&d_status, std::string("not_initialized"), "status", "Status string for diagnostics."))
{
    this->f_listening.setValue(true);
}

void WireWallForceMonitor::init()
{
    BaseObject::init();
    if (!l_collisionMechanicalObject)
    {
        setUnavailable("collisionMechanicalObject link is not set");
        return;
    }
    if (!l_wireMechanicalObject)
    {
        d_status.setValue("wireMechanicalObject link not set, using collision points");
    }
    if (!l_vesselMechanicalObject)
    {
        setUnavailable("vesselMechanicalObject link is not set");
        return;
    }
    if (!l_vesselTopology)
    {
        setUnavailable("vesselTopology link is not set");
        return;
    }
    d_source.setValue("passive_monitor");
    if (d_status.getValue() == "not_initialized")
    {
        d_status.setValue("ready");
    }
}

void WireWallForceMonitor::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        return;
    }
    updateTelemetry();
}

bool WireWallForceMonitor::readStatePositions3(
    const MechanicalState* state,
    Vec3List& out) const
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

bool WireWallForceMonitor::readStateForces3(
    const MechanicalState* state,
    sofa::core::ConstVecDerivId forceId,
    Vec3List& out) const
{
    out.clear();
    if (state == nullptr)
    {
        return false;
    }
    const auto n = state->getSize();
    const auto dim = state->getDerivDimension();
    if (n == 0 || dim < 3)
    {
        return false;
    }
    const auto scalarCount = static_cast<sofa::Size>(n) * static_cast<sofa::Size>(dim);
    std::vector<::SReal> raw(scalarCount, static_cast<::SReal>(0.0));
    try
    {
        state->copyToBuffer(raw.data(), forceId, static_cast<unsigned int>(scalarCount));
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

bool WireWallForceMonitor::updateWallTriangleCentroids()
{
    auto* vesselState = l_vesselMechanicalObject.get();
    auto* vesselTopo = l_vesselTopology.get();
    if (vesselState == nullptr || vesselTopo == nullptr)
    {
        return false;
    }

    Vec3List vesselPos;
    if (!readStatePositions3(vesselState, vesselPos))
    {
        return false;
    }
    auto& triangles = vesselTopo->getTriangles();
    if (triangles.empty())
    {
        return false;
    }

    m_wallTriangleCentroids.assign(triangles.size(), Vec3(0.0, 0.0, 0.0));
    for (sofa::Size triId = 0; triId < triangles.size(); ++triId)
    {
        const auto& tri = triangles[triId];
        const auto i0 = static_cast<sofa::Size>(tri[0]);
        const auto i1 = static_cast<sofa::Size>(tri[1]);
        const auto i2 = static_cast<sofa::Size>(tri[2]);
        if (i0 >= vesselPos.size() || i1 >= vesselPos.size() || i2 >= vesselPos.size())
        {
            return false;
        }
        m_wallTriangleCentroids[triId] =
            (vesselPos[i0] + vesselPos[i1] + vesselPos[i2]) / 3.0;
    }
    m_cachedWallTriangleCount = m_wallTriangleCentroids.size();
    return true;
}

unsigned int WireWallForceMonitor::findNearestWallTriangle(const Vec3& p) const
{
    if (m_wallTriangleCentroids.empty())
    {
        return 0U;
    }
    sofa::Size bestIdx = 0;
    double bestD2 = std::numeric_limits<double>::max();
    for (sofa::Size i = 0; i < m_wallTriangleCentroids.size(); ++i)
    {
        const auto d = p - m_wallTriangleCentroids[i];
        const auto d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestIdx = i;
        }
    }
    return static_cast<unsigned int>(bestIdx);
}

void WireWallForceMonitor::updateTelemetry()
{
    auto* collisionState = l_collisionMechanicalObject.get();
    if (collisionState == nullptr)
    {
        setUnavailable("collisionMechanicalObject link resolves to null");
        return;
    }
    if (l_vesselMechanicalObject.get() == nullptr)
    {
        setUnavailable("vesselMechanicalObject link resolves to null");
        return;
    }
    if (l_vesselTopology.get() == nullptr)
    {
        setUnavailable("vesselTopology link resolves to null");
        return;
    }

    struct ForceCandidate
    {
        std::string name;
        Vec3List forces;
        Vec3List positions;
        double normSum {0.0};
    };

    sofa::type::vector<ForceCandidate> candidates;

    auto addCandidate =
        [&](const std::string& prefix, MechanicalState* state) {
            if (state == nullptr)
            {
                return;
            }

            Vec3List positions;
            if (!readStatePositions3(state, positions) || positions.empty())
            {
                return;
            }

            auto addByForceId = [&](sofa::core::ConstVecDerivId forceId, const std::string& forceName) {
                Vec3List forces;
                if (!readStateForces3(state, forceId, forces) || forces.empty())
                {
                    return;
                }
                const auto n = std::min(forces.size(), positions.size());
                if (n == 0)
                {
                    return;
                }
                ForceCandidate c;
                c.name = prefix + "." + forceName;
                c.forces.assign(
                    forces.begin(),
                    forces.begin() + static_cast<std::ptrdiff_t>(n));
                c.positions.assign(
                    positions.begin(),
                    positions.begin() + static_cast<std::ptrdiff_t>(n));
                c.normSum = std::accumulate(
                    c.forces.begin(),
                    c.forces.end(),
                    0.0,
                    [](double acc, const Vec3& v) { return acc + vecNorm(v); });
                candidates.push_back(std::move(c));
            };

            addByForceId(sofa::core::ConstVecDerivId::force(), "force");
            addByForceId(sofa::core::ConstVecDerivId::externalForce(), "externalForce");
        };

    addCandidate("collision", collisionState);
    addCandidate("wire", l_wireMechanicalObject.get());

    if (candidates.empty())
    {
        setUnavailable("no readable force/position vectors on collision/wire states");
        return;
    }

    auto bestIt = std::max_element(
        candidates.begin(),
        candidates.end(),
        [](const ForceCandidate& a, const ForceCandidate& b) {
            return a.normSum < b.normSum;
        });
    if (bestIt == candidates.end())
    {
        setUnavailable("failed to select force candidate");
        return;
    }

    const auto& forceSourceName = bestIt->name;
    const auto& pointForces = bestIt->forces;
    const auto& pointPos = bestIt->positions;
    const auto nPoints = std::min(pointForces.size(), pointPos.size());
    if (!updateWallTriangleCentroids())
    {
        setUnavailable("failed to compute vessel wall triangle centroids");
        return;
    }
    const auto wallSegmentCount = m_wallTriangleCentroids.size();
    Vec3List wallSegmentForces(wallSegmentCount, Vec3(0.0, 0.0, 0.0));

    const double eps = std::max(0.0, d_contactEpsilon.getValue());

    for (sofa::Size i = 0; i < nPoints; ++i)
    {
        const auto& f = pointForces[i];
        if (vecNorm(f) > eps && wallSegmentCount > 0)
        {
            const auto tri = findNearestWallTriangle(pointPos[i]);
            // Contact force on wall is opposite to force applied on wire point.
            wallSegmentForces[tri] -= f;
        }
    }

    Vec3List activeForces;
    IndexList activeIds;
    activeForces.reserve(wallSegmentCount);
    activeIds.reserve(wallSegmentCount);
    Vec3 total(0.0, 0.0, 0.0);
    for (sofa::Size triId = 0; triId < wallSegmentForces.size(); ++triId)
    {
        const auto& wf = wallSegmentForces[triId];
        total += wf;
        if (vecNorm(wf) > eps)
        {
            activeForces.push_back(wf);
            activeIds.push_back(static_cast<unsigned int>(triId));
        }
    }

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> contactForcesAcc(d_contactForceVectors);
    contactForcesAcc.wref() = activeForces;

    sofa::helper::WriteAccessor<sofa::Data<IndexList>> contactSegmentsAcc(d_contactSegmentIndices);
    contactSegmentsAcc.wref() = activeIds;

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> segmentForcesAcc(d_segmentForceVectors);
    segmentForcesAcc.wref() = wallSegmentForces;

    sofa::helper::WriteAccessor<sofa::Data<IndexList>> wallActiveIdsAcc(d_wallActiveSegmentIds);
    wallActiveIdsAcc.wref() = activeIds;

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> wallActiveForcesAcc(d_wallActiveSegmentForceVectors);
    wallActiveForcesAcc.wref() = activeForces;

    d_totalForceVector.setValue(total);
    d_wallTotalForceVector.setValue(total);
    d_totalForceNorm.setValue(vecNorm(total));
    d_contactCount.setValue(static_cast<unsigned int>(activeIds.size()));
    d_wallSegmentCount.setValue(static_cast<unsigned int>(wallSegmentCount));

    d_available.setValue(true);
    d_source.setValue("passive_monitor_wall_triangles");
    d_status.setValue(
        std::string("ok:") + forceSourceName + ":nearest_triangle_centroid"
        + ":norm_sum=" + std::to_string(bestIt->normSum));
}

void WireWallForceMonitor::setUnavailable(const std::string& reason)
{
    d_available.setValue(false);
    d_status.setValue(reason);
    d_totalForceVector.setValue(Vec3(0.0, 0.0, 0.0));
    d_wallTotalForceVector.setValue(Vec3(0.0, 0.0, 0.0));
    d_totalForceNorm.setValue(0.0);
    d_contactCount.setValue(0);
    d_wallSegmentCount.setValue(0);

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> contactForcesAcc(d_contactForceVectors);
    contactForcesAcc.wref().clear();

    sofa::helper::WriteAccessor<sofa::Data<IndexList>> contactSegmentsAcc(d_contactSegmentIndices);
    contactSegmentsAcc.wref().clear();

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> segmentForcesAcc(d_segmentForceVectors);
    segmentForcesAcc.wref().clear();

    sofa::helper::WriteAccessor<sofa::Data<IndexList>> wallActiveIdsAcc(d_wallActiveSegmentIds);
    wallActiveIdsAcc.wref().clear();

    sofa::helper::WriteAccessor<sofa::Data<Vec3List>> wallActiveForcesAcc(d_wallActiveSegmentForceVectors);
    wallActiveForcesAcc.wref().clear();
}

} // namespace sofa::component::monitor

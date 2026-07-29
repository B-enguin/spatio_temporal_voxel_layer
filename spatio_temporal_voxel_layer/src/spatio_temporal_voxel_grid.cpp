/*********************************************************************
 *
 * Software License Agreement
 *
 *  Copyright (c) 2018, Simbe Robotics, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Simbe Robotics, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Steve Macenski (steven.macenski@simberobotics.com)
 *********************************************************************/

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "nav2_costmap_2d/cost_values.hpp"
#include "spatio_temporal_voxel_layer/spatio_temporal_voxel_grid.hpp"

namespace volume_grid
{

namespace
{
constexpr float MIN_OCCUPANCY = 0.0f;
constexpr float MAX_OCCUPANCY = 255.0f;
constexpr float OCCUPANCY_MARK_INCREMENT = 1.0f;
constexpr float OCCUPANCY_FRUSTUM_DECREMENT = 0.1f;
constexpr float MIN_OCCUPANCY_THRESHOLD = 5.0f;
constexpr float MIN_AFFORDANCE_MEAN = 0.0f;
constexpr float MAX_AFFORDANCE_MEAN = 1.0f;

inline float ClampOccupancy(const float value)
{
  return std::max(MIN_OCCUPANCY, std::min(MAX_OCCUPANCY, value));
}

inline float ClampAffordanceMean(const float value)
{
  return std::max(MIN_AFFORDANCE_MEAN, std::min(MAX_AFFORDANCE_MEAN, value));
}

float ComputeMaxAffordanceMean(
  const std::unordered_map<std::string, ObjectInstance::AlphaBetaPair> & affordances)
{
  float max_mean = MIN_AFFORDANCE_MEAN;
  for (const auto & affordance : affordances) {
    const float alpha = affordance.second.first;
    const float beta = affordance.second.second;
    const float denominator = alpha + beta;
    if (denominator <= 0.0f) {
      continue;
    }
    max_mean = std::max(max_mean, ClampAffordanceMean(alpha / denominator));
  }
  return max_mean;
}

uint GetObjectCost(const uint32_t object_id, const std::unordered_map<uint32_t, ObjectInstance> & objects)
{
  if (object_id == 0u) {
    return nav2_costmap_2d::LETHAL_OBSTACLE;
  }

  float max_mean = MIN_AFFORDANCE_MEAN;
  const auto object_it = objects.find(object_id);
  if (object_it != objects.end()) {
    max_mean = object_it->second.max_affordance_mean;
  }

  const float scaled_cost = static_cast<float>(nav2_costmap_2d::LETHAL_OBSTACLE) *
    (1.0f - ClampAffordanceMean(max_mean));
  return static_cast<uint>(std::lround(ClampOccupancy(scaled_cost)));
}

inline uint32_t ToObjectId(const float value)
{
  if (!std::isfinite(value)) {
    return 0u;
  }

  const float clamped = std::max(
    0.0f, std::min(static_cast<float>(std::numeric_limits<uint32_t>::max()), value));
  return static_cast<uint32_t>(std::lround(clamped));
}

inline uint32_t ToObjectId(const double value)
{
  if (!std::isfinite(value)) {
    return 0u;
  }

  const double clamped = std::max(
    0.0, std::min(static_cast<double>(std::numeric_limits<uint32_t>::max()), value));
  return static_cast<uint32_t>(std::llround(clamped));
}

inline std::string ObjectName(const uint32_t id)
{
  return "object_" + std::to_string(id);
}

const sensor_msgs::msg::PointField * FindField(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::string & name)
{
  for (const auto & field : cloud.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

template<typename T>
T ReadFieldValue(const sensor_msgs::msg::PointCloud2 & cloud, const std::size_t offset)
{
  T value{};
  if (offset + sizeof(T) <= cloud.data.size()) {
    std::memcpy(&value, &cloud.data[offset], sizeof(T));
  }
  return value;
}

uint32_t ReadObjectId(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const sensor_msgs::msg::PointField * field,
  const std::size_t point_index)
{
  if (!field) {
    return 0u;
  }

  const std::size_t offset = (point_index * cloud.point_step) + field->offset;
  if (offset >= cloud.data.size()) {
    return 0u;
  }

  switch (field->datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return ToObjectId(static_cast<double>(ReadFieldValue<int8_t>(cloud, offset)));
    case sensor_msgs::msg::PointField::UINT8:
      return static_cast<uint32_t>(ReadFieldValue<uint8_t>(cloud, offset));
    case sensor_msgs::msg::PointField::INT16:
      return ToObjectId(static_cast<double>(ReadFieldValue<int16_t>(cloud, offset)));
    case sensor_msgs::msg::PointField::UINT16:
      return static_cast<uint32_t>(ReadFieldValue<uint16_t>(cloud, offset));
    case sensor_msgs::msg::PointField::INT32:
      return ToObjectId(static_cast<double>(ReadFieldValue<int32_t>(cloud, offset)));
    case sensor_msgs::msg::PointField::UINT32:
      return ReadFieldValue<uint32_t>(cloud, offset);
    case sensor_msgs::msg::PointField::FLOAT32:
      return ToObjectId(ReadFieldValue<float>(cloud, offset));
    case sensor_msgs::msg::PointField::FLOAT64:
      return ToObjectId(ReadFieldValue<double>(cloud, offset));
    default:
      return 0u;
  }
}
}  // namespace

/*****************************************************************************/
SpatioTemporalVoxelGrid::SpatioTemporalVoxelGrid(
  rclcpp::Clock::SharedPtr clock,
  const float & voxel_size, const double & background_value,
  const int & decay_model, const double & voxel_decay, const bool & pub_voxels)
: _clock(clock), _decay_model(decay_model), _background_value(background_value),
  _voxel_size(voxel_size), _voxel_decay(voxel_decay), _pub_voxels(pub_voxels),
  _grid_points(std::make_unique<std::vector<geometry_msgs::msg::Point32>>()),
  _cost_map(new std::unordered_map<occupany_cell, uint>),
  _objects(std::unordered_map<uint32_t, ObjectInstance>())
/*****************************************************************************/
{
  this->InitializeGrid();
}

/*****************************************************************************/
SpatioTemporalVoxelGrid::~SpatioTemporalVoxelGrid(void)
/*****************************************************************************/
{
  // pcl pointclouds free themselves
  if (_cost_map) {
    delete _cost_map;
  }
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::InitializeGrid(void)
/*****************************************************************************/
{
  // initialize the OpenVDB Grid volume
  openvdb::initialize();

  // make it default to background value
  _grid = openvdb::Vec3fGrid::create(
    openvdb::Vec3f(static_cast<float>(_background_value), 0.0f, 0.0f));

  // setup scale and tranform
  openvdb::Mat4d m = openvdb::Mat4d::identity();
  m.preScale(openvdb::Vec3d(_voxel_size, _voxel_size, _voxel_size));
  m.preTranslate(openvdb::Vec3d(0, 0, 0));
  m.preRotate(openvdb::math::Z_AXIS, 0);

  // setup transform and other metadata
  _grid->setTransform(openvdb::math::Transform::createLinearTransform(m));
  _grid->setName("SpatioTemporalVoxelLayer");
  _grid->insertMeta("Voxel Size", openvdb::FloatMetadata(_voxel_size));
  _grid->setGridClass(openvdb::GRID_LEVEL_SET);
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::ClearFrustums(
  const std::vector<observation::MeasurementReading> & clearing_readings,
  std::unordered_set<occupany_cell> & cleared_cells)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  // accelerate the decay of voxels interior to the frustum
  if (this->IsGridEmpty()) {
    _grid_points->clear();
    _cost_map->clear();
    return;
  }

  _grid_points->clear();
  _cost_map->clear();

  std::vector<frustum_model> obs_frustums;

  if (clearing_readings.size() == 0) {
    TemporalClearAndGenerateCostmap(obs_frustums, cleared_cells);
    return;
  }

  obs_frustums.reserve(clearing_readings.size());

  std::vector<observation::MeasurementReading>::const_iterator it =
    clearing_readings.begin();
  for (; it != clearing_readings.end(); ++it) {
    geometry::Frustum * frustum = nullptr;
    if (it->_model_type == DEPTH_CAMERA) {
      frustum = new geometry::DepthCameraFrustum(
        it->_vertical_fov_in_rad,
        it->_horizontal_fov_in_rad, it->_min_z_in_m, it->_max_z_in_m);
    } else if (it->_model_type == THREE_DIMENSIONAL_LIDAR) {
      frustum = new geometry::ThreeDimensionalLidarFrustum(
        it->_vertical_fov_in_rad, it->_vertical_fov_padding_in_m,
        it->_horizontal_fov_in_rad, it->_min_z_in_m, it->_max_z_in_m);
    } else {
      // add else if statement for each implemented model
      delete frustum;
      continue;
    }

    frustum->SetPosition(it->_origin);
    frustum->SetOrientation(it->_orientation);
    frustum->TransformModel();
    obs_frustums.emplace_back(frustum, it->_decay_acceleration);
  }
  TemporalClearAndGenerateCostmap(obs_frustums, cleared_cells);
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::TemporalClearAndGenerateCostmap(
  std::vector<frustum_model> & frustums,
  std::unordered_set<occupany_cell> & cleared_cells)
/*****************************************************************************/
{
  // sample time once for all clearing readings
  const double cur_time = _clock->now().seconds();
  openvdb::Vec3fGrid::Accessor accessor = _grid->getAccessor();

  // check each point in the grid for inclusion in a frustum
  openvdb::Vec3fGrid::ValueOnCIter cit_grid = _grid->cbeginValueOn();
  for (; cit_grid.test(); ++cit_grid) {
    const openvdb::Coord pt_index(cit_grid.getCoord());
    const openvdb::Vec3d pose_world = this->IndexToWorld(pt_index);

    std::vector<frustum_model>::iterator frustum_it = frustums.begin();
    bool frustum_cycle = false;
    bool cleared_point = false;

    const double time_since_marking = cur_time - cit_grid.getValue()[0];
    const double base_duration_to_decay = GetTemporalClearingDuration(
      time_since_marking);

    for (; frustum_it != frustums.end(); ++frustum_it) {
      if (!frustum_it->frustum) {
        continue;
      }

      if (frustum_it->frustum->IsInside(pose_world) ) {
        frustum_cycle = true;
        openvdb::Vec3f updated_value = accessor.getValue(pt_index);
        updated_value[2] = ClampOccupancy(updated_value[2] - OCCUPANCY_FRUSTUM_DECREMENT);
        accessor.setValueOn(pt_index, updated_value);

        const double frustum_acceleration = GetFrustumAcceleration(
          time_since_marking, frustum_it->accel_factor);

        const double time_until_decay = base_duration_to_decay -
          frustum_acceleration;
        if (time_until_decay < 0.) {
          // expired by acceleration
          cleared_point = true;
          if (!this->ClearGridPoint(pt_index)) {
            std::cout << "Failed to clear point." << std::endl;
          }
          break;
        } else {
          const double updated_mark = cit_grid.getValue()[0] -
            frustum_acceleration;
          if (!this->MarkGridPoint(pt_index, updated_mark)) {
            std::cout << "Failed to update mark." << std::endl;
          }
          break;
        }
      }
    }

    // if not inside any, check against nominal decay model
    if (!frustum_cycle) {
      if (base_duration_to_decay < 0.) {
        // expired by temporal clearing
        cleared_point = true;
        if (!this->ClearGridPoint(pt_index)) {
          std::cout << "Failed to clear point." << std::endl;
        }
      }
    }

    if (cleared_point)
    {
      cleared_cells.insert(occupany_cell(pose_world[0], pose_world[1]));
    }
  }

  // free memory taken by expired voxels
  _grid->pruneGrid();

  const float occupancy_threshold = GetOccupancyThreshold();
  for (openvdb::Vec3fGrid::ValueOnCIter it = _grid->cbeginValueOn(); it.test(); ++it) {
    PopulateCostmapAndPointcloud(it.getCoord(), occupancy_threshold);
  }
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::PopulateCostmapAndPointcloud(
  const openvdb::Coord & pt, const float occupancy_threshold)
/*****************************************************************************/
{
  // add pt to the pointcloud and costmap
  openvdb::Vec3fGrid::Accessor accessor = _grid->getAccessor();
  const openvdb::Vec3f value = accessor.getValue(pt);
  if (value[2] <= occupancy_threshold) {
    return;
  }

  openvdb::Vec3d pose_world = this->IndexToWorld(pt);

  if (_pub_voxels) {
    geometry_msgs::msg::Point32 point;
    point.x = pose_world[0];
    point.y = pose_world[1];
    point.z = pose_world[2];
    _grid_points->push_back(point);
  }

  const uint32_t object_id = ToObjectId(value[1]);
  const uint voxel_cost = GetObjectCost(object_id, _objects);
  const occupany_cell map_cell(pose_world[0], pose_world[1]);

  std::unordered_map<occupany_cell, uint>::iterator cell;
  cell = _cost_map->find(map_cell);
  if (cell != _cost_map->end()) {
    cell->second = std::max(cell->second, voxel_cost);
  } else {
    _cost_map->insert(std::make_pair(map_cell, voxel_cost));
  }
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::Mark(
  const std::vector<observation::MeasurementReading> & marking_readings)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  // mark the grid
  if (marking_readings.size() > 0) {
    for (uint i = 0; i != marking_readings.size(); i++) {
      (*this)(marking_readings.at(i));
    }
  }
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::operator()(
  const observation::MeasurementReading & obs)
/*****************************************************************************/
{
  if (obs._marking) {
    float mark_range_2 = obs._obstacle_range_in_m * obs._obstacle_range_in_m;
    const double cur_time = _clock->now().seconds();

    const sensor_msgs::msg::PointCloud2 & cloud = *(obs._cloud);
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

    const sensor_msgs::msg::PointField * object_id_field = FindField(cloud, "id");
    if (!object_id_field) {
      object_id_field = FindField(cloud, "object_id");
    }

    std::size_t point_index = 0u;
    for (; iter_x != iter_x.end();
      ++iter_x, ++iter_y, ++iter_z, ++point_index)
    {
      const uint32_t object_id = ReadObjectId(cloud, object_id_field, point_index);
      const float object_id_value = static_cast<float>(object_id);

      float distance_2 =
        (*iter_x - obs._origin.x) * (*iter_x - obs._origin.x) +
        (*iter_y - obs._origin.y) * (*iter_y - obs._origin.y) +
        (*iter_z - obs._origin.z) * (*iter_z - obs._origin.z);
      if (distance_2 > mark_range_2 || distance_2 < 0.0001) {
        continue;
      }

      double x = *iter_x < 0 ? *iter_x - _voxel_size : *iter_x;
      double y = *iter_y < 0 ? *iter_y - _voxel_size : *iter_y;
      double z = *iter_y < 0 ? *iter_z - _voxel_size : *iter_z;

      openvdb::Vec3d mark_grid(this->WorldToIndex(
          openvdb::Vec3d(x, y, z)));

      const openvdb::Coord coord(mark_grid[0], mark_grid[1], mark_grid[2]);
      if (!this->MarkGridPoint(coord, cur_time))
      {
        std::cout << "Failed to mark point." << std::endl;
        continue;
      }

      openvdb::Vec3fGrid::Accessor accessor = _grid->getAccessor();
      openvdb::Vec3f value = accessor.getValue(coord);
      value[2] = ClampOccupancy(value[2] + OCCUPANCY_MARK_INCREMENT);

      // A zero incoming id means "unknown"; keep the voxel's current object id.
      if (object_id != 0u) {
        value[1] = object_id_value;

        ObjectInstance & object = _objects[object_id];
        if (object.name.empty()) {
          object.id = object_id;
          object.name = ObjectName(object_id);
          object.max_affordance_mean = MIN_AFFORDANCE_MEAN;
        }
      }
      accessor.setValueOn(coord, value);
    }
  }
}

/*****************************************************************************/
std::unordered_map<occupany_cell, uint> *
SpatioTemporalVoxelGrid::GetFlattenedCostmap()
/*****************************************************************************/
{
  return _cost_map;
}

/*****************************************************************************/
double SpatioTemporalVoxelGrid::GetTemporalClearingDuration(
  const double & time_delta)
/*****************************************************************************/
{
  // use configurable model to get desired decay time
  if (_decay_model == 0) {  // Linear
    return _voxel_decay - time_delta;
  } else if (_decay_model == 1) {  // Exponential
    return _voxel_decay * std::exp(-time_delta);
  }
  return _voxel_decay;  // PERSISTENT
}

/*****************************************************************************/
double SpatioTemporalVoxelGrid::GetFrustumAcceleration(
  const double & time_delta, const double & acceleration_factor)
/*****************************************************************************/
{
  const double acceleration = 1. / 6. * acceleration_factor *
    (time_delta * time_delta * time_delta);
  return acceleration;
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::GetOccupancyPointCloud(
  std::unique_ptr<sensor_msgs::msg::PointCloud2> & pc2)
/*****************************************************************************/
{
  // convert the active occupied VDB voxels into a PointCloud2
  const float occupancy_threshold = GetOccupancyThreshold();
  std::vector<std::pair<openvdb::Vec3d, float>> points;
  points.reserve(_grid->activeVoxelCount());

  for (openvdb::Vec3fGrid::ValueOnCIter it = _grid->cbeginValueOn(); it.test(); ++it) {
    const openvdb::Vec3f value = it.getValue();
    if (value[2] <= occupancy_threshold) {
      continue;
    }
    points.emplace_back(IndexToWorld(it.getCoord()), value[1]);
  }

  pc2->width = points.size();
  pc2->height = 1;
  pc2->is_dense = true;
  pc2->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier modifier(*pc2);

  modifier.setPointCloud2Fields(
    4,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "id", 1, sensor_msgs::msg::PointField::FLOAT32);
  modifier.resize(points.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(*pc2, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*pc2, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*pc2, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_id(*pc2, "id");

  for (const auto & pt : points)
  {
    *iter_x = static_cast<float>(pt.first[0]);
    *iter_y = static_cast<float>(pt.first[1]);
    *iter_z = static_cast<float>(pt.first[2]);
    *iter_id = pt.second;
    ++iter_x; ++iter_y; ++iter_z; ++iter_id;
  }
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::ResetGrid(void)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  // clear the voxel grid
  try {
    _grid->clear();
    if (this->IsGridEmpty()) {
      return true;
    }
  } catch (...) {
    std::cout << "Failed to reset costmap, please try again." << std::endl;
  }
  return false;
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::ResetGridArea(
  const occupany_cell & start, const occupany_cell & end, bool invert_area)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  openvdb::Vec3fGrid::ValueOnCIter cit_grid = _grid->cbeginValueOn();
  for (cit_grid; cit_grid.test(); ++cit_grid)
  {
    const openvdb::Coord pt_index(cit_grid.getCoord());
    const openvdb::Vec3d pose_world = this->IndexToWorld(pt_index);

    const bool in_x_range = pose_world.x() > start.x && pose_world.x() < end.x;
    const bool in_y_range = pose_world.y() > start.y && pose_world.y() < end.y;
    const bool in_range = in_x_range && in_y_range;

    if(in_range != invert_area)
    {
      ClearGridPoint(pt_index);
    }
  }
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::MarkGridPoint(
  const openvdb::Coord & pt, const double & value) const
/*****************************************************************************/
{
  // marking the OpenVDB set
  openvdb::Vec3fGrid::Accessor accessor = _grid->getAccessor();

  const openvdb::Vec3f existing =
    accessor.isValueOn(pt) ? accessor.getValue(pt) :
    openvdb::Vec3f(static_cast<float>(_background_value), 0.0f, 0.0f);

  accessor.setValueOn(
    pt,
    openvdb::Vec3f(static_cast<float>(value), existing[1], existing[2]));
  return accessor.getValue(pt)[0] == static_cast<float>(value);
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::ClearGridPoint(const openvdb::Coord & pt) const
/*****************************************************************************/
{
  // clearing the OpenVDB set
  openvdb::Vec3fGrid::Accessor accessor = _grid->getAccessor();

  if (accessor.isValueOn(pt)) {
    accessor.setValueOff(
      pt, openvdb::Vec3f(static_cast<float>(_background_value), 0.0f, 0.0f));
  }
  return !accessor.isValueOn(pt);
}

/*****************************************************************************/
openvdb::Vec3d SpatioTemporalVoxelGrid::IndexToWorld(
  const openvdb::Coord & coord) const
/*****************************************************************************/
{
  // Applies tranform stored in getTransform.
  openvdb::Vec3d pose_world =  _grid->indexToWorld(coord);

  // Using the center for world coordinate
  const double & center_offset = _voxel_size / 2.0;
  pose_world[0] += center_offset;
  pose_world[1] += center_offset;
  pose_world[2] += center_offset;

  return pose_world;
}

/*****************************************************************************/
openvdb::Vec3d SpatioTemporalVoxelGrid::WorldToIndex(
  const openvdb::Vec3d & vec) const
/*****************************************************************************/
{
  // Applies inverse tranform stored in getTransform.
  return _grid->worldToIndex(vec);
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::IsGridEmpty(void) const
/*****************************************************************************/
{
  // Returns grid's population status
  return _grid->empty();
}

/*****************************************************************************/
float SpatioTemporalVoxelGrid::GetOccupancyThreshold(void) const
/*****************************************************************************/
{
  float max_occupancy = MIN_OCCUPANCY;
  for (openvdb::Vec3fGrid::ValueOnCIter it = _grid->cbeginValueOn(); it.test(); ++it) {
    max_occupancy = std::max(max_occupancy, ClampOccupancy(it.getValue()[2]));
  }
  return std::max(MIN_OCCUPANCY_THRESHOLD, max_occupancy * 0.5f);
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::SaveGrid(
  const std::string & file_name, double & map_size_bytes)
/*****************************************************************************/
{
  try {
    openvdb::io::File file(file_name + ".vdb");
    openvdb::GridPtrVec grids = {_grid};
    file.write(grids);
    file.close();
    map_size_bytes = _grid->memUsage();
    return true;
  } catch (...) {
    map_size_bytes = 0.;
    return false;
  }
  return false;  // best offense is a good defense
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::AddObject(
  const uint32_t id, const std::string & name,
  const std::unordered_map<std::string, ObjectInstance::AlphaBetaPair> & affordances)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  if (id == 0u) {
    return false;
  }

  ObjectInstance object;
  object.id = id;
  object.name = name;
  object.alpha_beta_values = affordances;
  object.max_affordance_mean = ComputeMaxAffordanceMean(object.alpha_beta_values);
  _objects[id] = object;
  return true;
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::GetObject(
  const uint32_t id, ObjectInstance & object) const
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  const auto object_it = _objects.find(id);
  if (object_it == _objects.end()) {
    return false;
  }

  object = object_it->second;
  return true;
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::UpdateObject(
  const uint32_t id, const std::vector<std::string> & affordance_names,
  const std::vector<float> & alphas, const std::vector<float> & betas)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  if (affordance_names.size() != alphas.size() || affordance_names.size() != betas.size()) {
    return false;
  }

  const auto object_it = _objects.find(id);
  if (object_it == _objects.end()) {
    return false;
  }

  for (std::size_t i = 0; i < affordance_names.size(); ++i) {
    object_it->second.alpha_beta_values[affordance_names[i]] =
      ObjectInstance::AlphaBetaPair(alphas[i], betas[i]);
  }
  object_it->second.max_affordance_mean =
    ComputeMaxAffordanceMean(object_it->second.alpha_beta_values);
  return true;
}

}  // namespace volume_grid

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
#include <cstring>
#include <cmath>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>

#include "spatio_temporal_voxel_layer/spatio_temporal_voxel_grid.hpp"

namespace volume_grid
{

namespace
{
inline uint64_t ToIdKey(const float value)
{
  const float clamped = std::max(0.0f, std::min(static_cast<float>(std::numeric_limits<uint64_t>::max()), value));
  return static_cast<uint64_t>(std::llround(static_cast<double>(clamped)));
}

inline uint64_t ToIdKey(const double value)
{
  const double clamped = std::max(0.0, std::min(static_cast<double>(std::numeric_limits<uint64_t>::max()), value));
  return static_cast<uint64_t>(std::llround(clamped));
}

inline bool IsAffordanceField(const std::string & name)
{
  return name.rfind("affordance", 0) == 0 && name != "affordance_count";
}

inline int ParseAffordanceIndex(const std::string & name, const int fallback)
{
  std::size_t end = name.size();
  std::size_t start = end;
  while (start > 0 && std::isdigit(static_cast<unsigned char>(name[start - 1]))) {
    --start;
  }
  if (start == end) {
    return fallback;
  }
  return std::stoi(name.substr(start, end - start));
}

inline std::vector<std::string> GetOrderedAffordanceFields(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  std::vector<std::pair<int, std::string>> indexed_fields;
  int fallback = 1000;
  for (const auto & field : cloud.fields) {
    if (!IsAffordanceField(field.name)) {
      continue;
    }
    indexed_fields.emplace_back(ParseAffordanceIndex(field.name, fallback++), field.name);
  }

  std::sort(
    indexed_fields.begin(), indexed_fields.end(),
    [](const auto & a, const auto & b) {
      return (a.first == b.first) ? (a.second < b.second) : (a.first < b.first);
    });

  std::vector<std::string> out;
  out.reserve(indexed_fields.size());
  for (const auto & it : indexed_fields) {
    out.push_back(it.second);
  }
  return out;
}
}  // namespace

/*****************************************************************************/
SpatioTemporalVoxelGrid::SpatioTemporalVoxelGrid(
  rclcpp::Clock::SharedPtr clock,
  const float & voxel_size, const double & background_value,
  const int & decay_model, const double & voxel_decay, const bool & pub_voxels,
  const uint32_t & max_affordances, const double & afforded_factor)
: _clock(clock), _decay_model(decay_model), _background_value(background_value),
  _voxel_size(voxel_size), _voxel_decay(voxel_decay), _pub_voxels(pub_voxels),
  _grid_points(std::make_unique<std::vector<geometry_msgs::msg::Point32>>()),
  _cost_map(new std::unordered_map<occupany_cell, std::pair<uint, float>>),
  _affordance_map(std::unordered_map<uint64_t, std::vector<double>>()),
  _max_affordances(max_affordances),
  _afforded_factor(std::max(0.0, afforded_factor))
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
  _grid = openvdb::Vec3dGrid::create(openvdb::Vec3d(_background_value, 0.0, 0.0));

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

  // check each point in the grid for inclusion in a frustum
  openvdb::Vec3dGrid::ValueOnCIter cit_grid = _grid->cbeginValueOn();
  for (; cit_grid.test(); ++cit_grid) {
    const openvdb::Coord pt_index(cit_grid.getCoord());
    const openvdb::Vec3d pose_world = this->IndexToWorld(pt_index);

    std::vector<frustum_model>::iterator frustum_it = frustums.begin();
    bool frustum_cycle = false;
    bool cleared_point = false;

    const double time_since_marking = cur_time - cit_grid.getValue()[0];
    double base_duration_to_decay = GetTemporalClearingDuration(time_since_marking);
    if (HasAffordances(cit_grid.getValue())) {
      base_duration_to_decay *= _afforded_factor;
    }

    for (; frustum_it != frustums.end(); ++frustum_it) {
      if (!frustum_it->frustum) {
        continue;
      }

      if (frustum_it->frustum->IsInside(pose_world) ) {
        frustum_cycle = true;

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
    } else {
      // if here, we can add to costmap and PC2
      PopulateCostmapAndPointcloud(pt_index);
    }
  }

  // free memory taken by expired voxels
  _grid->pruneGrid();
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::HasAffordances(const openvdb::Vec3d & value) const
/*****************************************************************************/
{
  const uint64_t id = ToIdKey(value[1]);
  if (id == 0u) {
    return false;
  }

  const auto affordance_it = _affordance_map.find(id);
  return affordance_it != _affordance_map.end() && !affordance_it->second.empty();
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::PopulateCostmapAndPointcloud(
  const openvdb::Coord & pt)
/*****************************************************************************/
{
  // add pt to the pointcloud and costmap
  openvdb::Vec3d pose_world = this->IndexToWorld(pt);

  if (_pub_voxels) {
    geometry_msgs::msg::Point32 point;
    point.x = pose_world[0];
    point.y = pose_world[1];
    point.z = pose_world[2];
    _grid_points->push_back(point);
  }

  openvdb::Vec3dGrid::Accessor accessor = _grid->getAccessor();
  openvdb::Vec3d value = accessor.getValue(pt);

  const uint64_t id_key = ToIdKey(value[1]);
  float affordance = 0.0f;
  const auto id_it = _affordance_map.find(id_key);
  if (id_it != _affordance_map.end()) {
    for (const auto a : id_it->second) {
      affordance = std::max(affordance, static_cast<float>(a));
    }
  }

  std::unordered_map<occupany_cell, std::pair<uint, float>>::iterator cell;
  cell = _cost_map->find(occupany_cell(pose_world[0], pose_world[1]));
  if (cell != _cost_map->end()) {
    cell->second.first += 1;

    if (affordance > cell->second.second) {
      cell->second.second = affordance;
    }
  } else {
    _cost_map->insert(
      std::make_pair(
        occupany_cell(pose_world[0], pose_world[1]), std::make_pair(1, affordance)));
  }

  // if (id_key != 0u && !affordances.empty()) {
  //   auto & stored = _affordance_map[id_key];
  //   if (stored.size() < affordances.size()) {
  //     stored.resize(affordances.size(), 0.0);
  //   }
  //   for (std::size_t i = 0; i < affordances.size(); ++i) {
  //     stored[i] = std::max(stored[i], affordances[i]);
  //   }
  // }
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

    bool has_id = false;
    std::string id_field_name;
    for (const auto & field : cloud.fields) {
      if (field.name == "id") {
        has_id = true;
        id_field_name = field.name;
        break;
      }
    }

    std::vector<std::string> affordance_fields = GetOrderedAffordanceFields(cloud);
    if (_max_affordances > 0u && affordance_fields.size() > _max_affordances) {
      std::cout << "Received " << affordance_fields.size()
                << " affordance fields; using first " << _max_affordances << "." << std::endl;
      affordance_fields.resize(_max_affordances);
    }

    std::vector<sensor_msgs::PointCloud2ConstIterator<float>> iter_affordances;
    iter_affordances.reserve(affordance_fields.size());
    for (const auto & name : affordance_fields) {
      iter_affordances.emplace_back(cloud, name);
    }

    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iter_id;
    if (has_id) {
      iter_id =
        std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(cloud, id_field_name);
    }

    openvdb::Vec3dGrid::Accessor accessor = _grid->getAccessor();

    auto update_voxel =
      [&](const float px, const float py, const float pz,
      const uint64_t id_key, const std::vector<double> & affordances)
      {
        const float distance_2 =
          (px - obs._origin.x) * (px - obs._origin.x) +
          (py - obs._origin.y) * (py - obs._origin.y) +
          (pz - obs._origin.z) * (pz - obs._origin.z);

        if (distance_2 > mark_range_2 || distance_2 < 0.0001f) {
          return;
        }

        const double x = px < 0 ? px - _voxel_size : px;
        const double y = py < 0 ? py - _voxel_size : py;
        const double z = pz < 0 ? pz - _voxel_size : pz;

        const openvdb::Vec3d mark_grid = this->WorldToIndex(openvdb::Vec3d(x, y, z));
        const openvdb::Coord coord(mark_grid[0], mark_grid[1], mark_grid[2]);

        openvdb::Vec3d value = accessor.isValueOn(coord) ?
          accessor.getValue(coord) :
          openvdb::Vec3d(_background_value, 0.0, 0.0);

        // Use id_key is non-zero, otherwise preserve existing id if present
        value[0] = cur_time;
        if (id_key != 0u) {
          value[1] = static_cast<double>(id_key);
          value[2] = 0.0;                          // reserved placeholder
        }
        
        accessor.setValueOn(coord, value);

        if (id_key != 0u && !affordances.empty()) {
          auto & stored = _affordance_map[id_key];
          if (stored.size() < affordances.size()) {
            stored.resize(affordances.size(), 0.0);
          }
          for (std::size_t i = 0; i < affordances.size(); ++i) {
            stored[i] = std::max(stored[i], affordances[i]);
          }
        }
      } ;

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      uint64_t id_value = 0u;
      if (iter_id) {
        id_value = ToIdKey(**iter_id);
        ++(*iter_id);
      }

      std::vector<double> point_affordances;
      point_affordances.reserve(iter_affordances.size());
      for (auto & it : iter_affordances) {
        point_affordances.push_back(static_cast<double>(*it));
        ++it;
      }

      update_voxel(*iter_x, *iter_y, *iter_z, id_value, point_affordances);
    }
  }
}

/*****************************************************************************/
std::unordered_map<occupany_cell, std::pair<uint, float>> *
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
  // convert the grid points stored in a PointCloud2
  pc2->width = _grid_points->size();
  pc2->height = 1;
  pc2->is_dense = true;

  sensor_msgs::PointCloud2Modifier modifier(*pc2);

  modifier.setPointCloud2Fields(
    3,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  modifier.setPointCloud2FieldsByString(1, "xyz");

  sensor_msgs::PointCloud2Iterator<float> iter_x(*pc2, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*pc2, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*pc2, "z");

  for (std::vector<geometry_msgs::msg::Point32>::iterator it =
    _grid_points->begin();
    it != _grid_points->end(); ++it)
  {
    const geometry_msgs::msg::Point32 & pt = *it;
    *iter_x = pt.x;
    *iter_y = pt.y;
    *iter_z = pt.z;
    ++iter_x; ++iter_y; ++iter_z;
  }
}

/*****************************************************************************/
void SpatioTemporalVoxelGrid::GetSemanticPointCloud(
  std::unique_ptr<sensor_msgs::msg::PointCloud2> & pc2)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  std::vector<openvdb::Vec3d> points;
  std::vector<float> ids;
  std::vector<std::vector<float>> affordances_per_point;
  std::size_t max_affordances = 0;

  points.reserve(_grid->activeVoxelCount());
  ids.reserve(_grid->activeVoxelCount());
  affordances_per_point.reserve(_grid->activeVoxelCount());

  for (openvdb::Vec3dGrid::ValueOnCIter it = _grid->cbeginValueOn(); it.test(); ++it) {
    const openvdb::Vec3d value = it.getValue();
    const uint64_t id = ToIdKey(value[1]);

    std::vector<float> affordances;
    auto a_it = _affordance_map.find(id);
    if (a_it != _affordance_map.end()) {
      affordances.reserve(a_it->second.size());
      for (const auto v : a_it->second) {
        affordances.push_back(static_cast<float>(v));
      }
    }

    if (id == 0u && affordances.empty()) {
      continue;
    }

    max_affordances = std::max(max_affordances, affordances.size());
    points.push_back(IndexToWorld(it.getCoord()));
    ids.push_back(static_cast<float>(id));
    affordances_per_point.push_back(std::move(affordances));
  }

  pc2->width = points.size();
  pc2->height = 1;
  pc2->is_dense = true;
  pc2->is_bigendian = false;

  pc2->fields.clear();
  pc2->fields.reserve(5 + max_affordances);

  auto add_field = [&](const std::string & name, uint32_t offset) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    pc2->fields.push_back(f);
  };

  uint32_t offset = 0;
  add_field("x", offset); offset += 4;
  add_field("y", offset); offset += 4;
  add_field("z", offset); offset += 4;
  add_field("id", offset); offset += 4;
  add_field("affordance_count", offset); offset += 4;
  for (std::size_t i = 0; i < max_affordances; ++i) {
    add_field("affordance_" + std::to_string(i), offset);
    offset += 4;
  }

  pc2->point_step = offset;
  pc2->row_step = pc2->point_step * pc2->width;
  pc2->data.assign(pc2->row_step, 0u);

  auto write_f32 = [&](std::size_t base, uint32_t off, float v) {
    std::memcpy(&pc2->data[base + off], &v, sizeof(float));
  };

  for (std::size_t i = 0; i < points.size(); ++i) {
    const std::size_t base = i * pc2->point_step;
    write_f32(base, 0, static_cast<float>(points[i][0]));
    write_f32(base, 4, static_cast<float>(points[i][1]));
    write_f32(base, 8, static_cast<float>(points[i][2]));
    write_f32(base, 12, ids[i]);
    write_f32(base, 16, static_cast<float>(affordances_per_point[i].size()));
    for (std::size_t j = 0; j < affordances_per_point[i].size(); ++j) {
      write_f32(base, static_cast<uint32_t>(20 + (j * 4)), affordances_per_point[i][j]);
    }
  }
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::ResetGrid(void)
/*****************************************************************************/
{
  boost::unique_lock<boost::mutex> lock(_grid_lock);

  try {
    _grid->clear();
    _affordance_map.clear();
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

  openvdb::Vec3dGrid::ValueOnCIter cit_grid = _grid->cbeginValueOn();
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
  openvdb::Vec3dGrid::Accessor accessor = _grid->getAccessor();

  const openvdb::Vec3d existing =
    accessor.isValueOn(pt) ? accessor.getValue(pt)
                           : openvdb::Vec3d(_background_value, 0.0, 0.0);

  accessor.setValueOn(pt, openvdb::Vec3d(value, existing[1], existing[2]));
  return accessor.getValue(pt)[0] == value;
}

/*****************************************************************************/
bool SpatioTemporalVoxelGrid::ClearGridPoint(const openvdb::Coord & pt) const
/*****************************************************************************/
{
  // clearing the OpenVDB set
  openvdb::Vec3dGrid::Accessor accessor = _grid->getAccessor();

  if (accessor.isValueOn(pt)) {
    accessor.setValueOff(pt, openvdb::Vec3d(_background_value, 0.0, 0.0));
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

}  // namespace volume_grid

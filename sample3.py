import bpy
import random
import os
import math
import bmesh
from mathutils import Matrix, Vector
import bpy_extras.object_utils

# --- Configuration ---
OUTPUT_DIR = "D:/asunama/synthetic datasets/dataset"
NUM_IMAGES = 2

# CRITICAL CAMERA SETTINGS FOR 30M HEIGHT
CAMERA_HEIGHT = 30  # 30 meters from ground

# GROUND SIZE CALCULATION:
# Camera FOV at 30m with 4.6mm lens on 1/1.7" sensor:
# Horizontal FOV ≈ 81°, Visible width = 2 × tan(40.5°) × 30 ≈ 51m
# To ensure full coverage with safety margin, ground must be at least 60m × 60m
GROUND_SIZE = 60  # Increased from 40m to 60m to cover entire camera view

# Object size constraints in Blender units (assuming 1 unit = 1 meter)
MIN_OBJECT_SIZE = 0.3  # 30cm in meters
MAX_OBJECT_SIZE = 0.8  # 80cm in meters

HDRI_PATHS = [
    "D:/asunama/synthetic datasets/hdri/steinbach_field_4k.exr",
]

GROUND_TEXTURE_PATHS = [
    "D:/asunama/synthetic datasets/textures/rocky_terrain_02_diff_4k.jpg",
]

def clear_scene():
    """Removes all objects from the scene to start fresh."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera():
    """Sets up a camera at 30m altitude matching 1/1.7-inch Sony Starlight CMOS sensor."""
    # Add a camera at 30 meters altitude, directly above the center of the scene
    bpy.ops.object.camera_add(location=(0, 0, CAMERA_HEIGHT))
    camera = bpy.context.object

    # Set the rotation to point straight down along the negative Z-axis
    camera.rotation_euler = (0, 0, 0)

    # Use PERSPECTIVE camera for realistic lens simulation
    camera.data.type = 'PERSP'
    
    # Camera sensor specifications for 1/1.7-inch Sony Starlight CMOS
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 7.6  # mm (1/1.7-inch sensor width)
    camera.data.sensor_height = 4.28  # mm (for 16:9 aspect ratio to match 4K)
    
    # Actual focal length for 21mm equivalent on 1/1.7" sensor
    camera.data.lens = 4.6  # mm (actual focal length)
    
    # Set aperture to F2.8
    camera.data.dof.aperture_fstop = 2.8
    
    # Disable DOF for dataset generation (everything sharp)
    camera.data.dof.use_dof = False
    
    # Set render resolution to 640 x 640
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 640
    bpy.context.scene.render.resolution_percentage = 100
    
    # Set pixel aspect ratio to 1.0 (square pixels)
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0
    
    # Set the camera as active
    bpy.context.scene.camera = camera
    
    # Calculate and verify FOV coverage
    fov_horizontal = 2 * math.atan((camera.data.sensor_width / 2) / camera.data.lens)
    fov_horizontal_deg = math.degrees(fov_horizontal)
    visible_width = 2 * math.tan(fov_horizontal / 2) * CAMERA_HEIGHT
    visible_height = 2 * math.tan(math.atan((camera.data.sensor_height / 2) / camera.data.lens)) * CAMERA_HEIGHT
    
    print(f"\n{'='*60}")
    print(f"CAMERA SETUP - 30M ALTITUDE")
    print(f"{'='*60}")
    print(f"  Sensor: 1/1.7-inch ({camera.data.sensor_width}mm x {camera.data.sensor_height}mm)")
    print(f"  Focal Length: {camera.data.lens}mm (21mm equivalent)")
    print(f"  Aperture: F{camera.data.dof.aperture_fstop}")
    print(f"  Camera Height: {CAMERA_HEIGHT}m")
    print(f"  Horizontal FOV: {fov_horizontal_deg:.1f}°")
    print(f"  Visible Ground Area:")
    print(f"    - Width: {visible_width:.1f}m")
    print(f"    - Height: {visible_height:.1f}m")
    print(f"  Ground Texture Size: {GROUND_SIZE}m x {GROUND_SIZE}m")
    
    # Verify coverage
    if visible_width > GROUND_SIZE or visible_height > GROUND_SIZE:
        print(f"\n  ⚠️  WARNING: Camera view ({visible_width:.1f}m x {visible_height:.1f}m)")
        print(f"      exceeds ground texture ({GROUND_SIZE}m x {GROUND_SIZE}m)!")
        print(f"      Increase GROUND_SIZE to at least {max(visible_width, visible_height) + 5:.0f}m")
    else:
        print(f"\n  ✓ Camera view fully contained within ground texture")
        print(f"    Safety margin: {(GROUND_SIZE - visible_width) / 2:.1f}m on each side")
    print(f"{'='*60}\n")
    
    return camera

def setup_lighting_and_environment(hdri_path):
    """Sets up environment lighting using an HDRI with lighting settings from 15m configuration."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.world.use_nodes = True

    # Clear default nodes
    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links
    nodes.clear()

    # Create necessary nodes
    background = nodes.new(type='ShaderNodeBackground')
    environment_texture = nodes.new('ShaderNodeTexEnvironment')
    mapping_node = nodes.new(type='ShaderNodeMapping')
    tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
    output = nodes.new(type='ShaderNodeOutputWorld')

    # Load the HDRI image
    environment_texture.image = bpy.data.images.load(hdri_path)

    # Link the nodes to control the rotation
    links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], environment_texture.inputs['Vector'])
    links.new(environment_texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    # Randomize HDRI rotation using the Mapping node's Z-axis
    mapping_node.inputs['Rotation'].default_value[2] = random.uniform(0, 2 * math.pi)

    # Use exact lighting settings from 15m configuration (better lighting quality)
    background.inputs['Strength'].default_value = random.uniform(0.4, 1.2)

    # Add a Sun lamp with 15m configuration settings (produces better lighting)
    sun_height = random.uniform(18, 25)  # Sun slightly above camera
    bpy.ops.object.light_add(type='SUN', location=(random.uniform(-10, 10), random.uniform(-10, 10), sun_height))
    sun_light = bpy.context.object
    sun_light.data.energy = random.uniform(1.0, 3.0)  # Good lighting for detailed objects
    sun_light.rotation_euler = (math.radians(random.uniform(30, 70)), math.radians(random.uniform(-45, 45)), random.uniform(0, 2*math.pi))

def create_ground_plane(texture_path):
    """Creates a ground plane with a specified texture."""
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, enter_editmode=False, align='WORLD')
    ground_obj = bpy.context.object
    ground_obj.name = "Ground"

    mat = bpy.data.materials.new(name="GroundMaterial")
    ground_obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = 0, 0

    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(texture_path)
    texture_node.location = -300, 0

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = 200, 0

    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # Randomize texture scale/rotation for variety
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = -500, 0
    mapping_node.inputs['Scale'].default_value = (random.uniform(0.5, 2.0), random.uniform(0.5, 2.0), 1)
    mapping_node.inputs['Rotation'].default_value[2] = random.uniform(0, 2*math.pi)

    tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
    tex_coord_node.location = -700, 0

    links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])

def create_realistic_traffic_cone(cone_name):
    """Creates a detailed traffic cone based on your custom design."""
    
    original_selection = [obj for obj in bpy.context.selected_objects]
    original_active = bpy.context.active_object
    
    def create_cone_body():
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.20,
            radius2=0.04,
            depth=0.60,
            location=(0, 0, 0.30)
        )
        
        cone = bpy.context.active_object
        cone.name = f"{cone_name}_Body"
        
        bpy.context.view_layer.objects.active = cone
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=2)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return cone

    def create_cone_base():
        bpy.ops.mesh.primitive_cube_add(
            size=0.45,
            location=(0, 0, 0.035)
        )
        
        base = bpy.context.active_object
        base.name = f"{cone_name}_Base"
        base.scale[2] = 0.16
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        return base

    def create_cone_top():
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16,
            radius=0.015,
            depth=0.04,
            location=(0, 0, 0.64)
        )
        
        top = bpy.context.active_object
        top.name = f"{cone_name}_Top"
        
        return top

    def create_cone_materials():
        orange_mat = bpy.data.materials.new(name=f"Orange_Cone_{cone_name}")
        orange_mat.use_nodes = True
        
        nodes = orange_mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        
        links = orange_mat.node_tree.links
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (1.0, 0.3, 0.1, 1.0)
        principled.inputs['Roughness'].default_value = 0.3
        
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        silver_mat = bpy.data.materials.new(name=f"Reflective_Silver_{cone_name}")
        silver_mat.use_nodes = True
        
        nodes = silver_mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        
        links = silver_mat.node_tree.links
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.85, 1.0)
        principled.inputs['Metallic'].default_value = 0.9
        principled.inputs['Roughness'].default_value = 0.1
        
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        return orange_mat, silver_mat

    def apply_materials_and_stripe(cone_body, orange_mat, silver_mat):
        if len(cone_body.data.materials) == 0:
            cone_body.data.materials.append(orange_mat)
        
        cone_body.data.materials.append(silver_mat)
        
        bpy.context.view_layer.objects.active = cone_body
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        
        bm = bmesh.from_edit_mesh(cone_body.data)
        
        for face in bm.faces:
            face.select = False
        
        for face in bm.faces:
            face_center = face.calc_center_median()
            if 0.20 < face_center.z < 0.28:
                face.select = True
        
        bmesh.update_edit_mesh(cone_body.data)
        
        cone_body.active_material_index = 1
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')

    def add_text_to_cone():
        bpy.ops.object.text_add(location=(0, -0.3, 0.3))
        text_obj = bpy.context.active_object
        text_obj.name = f"{cone_name}_Text"
        
        text_obj.data.body = "TRAFFIC"
        text_obj.data.size = 0.03
        text_obj.data.extrude = 0.002
        text_obj.rotation_euler = (math.pi/2, 0, 0)
        
        bpy.ops.object.convert(target='MESH')
        
        return text_obj

    cone_body = create_cone_body()
    cone_base = create_cone_base()
    cone_top = create_cone_top()
    
    orange_mat, silver_mat = create_cone_materials()
    
    apply_materials_and_stripe(cone_body, orange_mat, silver_mat)
    
    cone_base.data.materials.append(orange_mat)
    cone_top.data.materials.append(orange_mat)
    
    text_obj = add_text_to_cone()
    text_obj.data.materials.append(orange_mat)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    
    cone_body.select_set(True)
    cone_base.select_set(True)
    cone_top.select_set(True)
    text_obj.select_set(True)
    
    bpy.context.view_layer.objects.active = cone_body
    bpy.ops.object.join()
    
    final_cone = bpy.context.active_object
    final_cone.name = cone_name
    bpy.ops.object.shade_smooth()
    
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in original_selection:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    if original_active and original_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = original_active
    
    return final_cone

def create_realistic_material(obj_type, material_name):
    """Creates realistic materials with procedural textures for different object types."""
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    principled_bsdf = nodes.get("Principled BSDF")
    output_node = nodes.get("Material Output")
    
    for node in nodes:
        if node not in [principled_bsdf, output_node]:
            nodes.remove(node)
    
    if obj_type == 'CUBE':
        material_type = random.choice(['cardboard', 'wood', 'metal', 'plastic'])
        
        if material_type == 'cardboard':
            base_color = [random.uniform(0.4, 0.7), random.uniform(0.25, 0.45), random.uniform(0.1, 0.25), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.7, 0.9)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif material_type == 'wood':
            wood_colors = [
                [0.4, 0.25, 0.15, 1.0],
                [0.6, 0.4, 0.25, 1.0],
                [0.7, 0.5, 0.35, 1.0]
            ]
            base_color = random.choice(wood_colors)
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 0.8)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif material_type == 'metal':
            base_color = [random.uniform(0.6, 0.8), random.uniform(0.6, 0.8), random.uniform(0.6, 0.8), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.2, 0.5)
            principled_bsdf.inputs['Metallic'].default_value = random.uniform(0.7, 0.9)
            
        else:
            base_color = [random.uniform(0.2, 0.9), random.uniform(0.2, 0.9), random.uniform(0.2, 0.9), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.4)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        noise_texture = nodes.new(type='ShaderNodeTexNoise')
        noise_texture.location = (-400, 0)
        noise_texture.inputs['Scale'].default_value = random.uniform(50, 150)
        noise_texture.inputs['Detail'].default_value = random.uniform(2, 8)
        noise_texture.inputs['Roughness'].default_value = random.uniform(0.4, 0.7)
        
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 0)
        color_ramp.color_ramp.elements[0].position = 0.4
        color_ramp.color_ramp.elements[1].position = 0.6
        
        links.new(noise_texture.outputs['Color'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Roughness'])
        
    elif obj_type == 'SPHERE':
        ball_type = random.choice(['soccer', 'basketball', 'beach_ball', 'rubber'])
        
        if ball_type == 'soccer':
            base_color = [random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.3, 0.6)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif ball_type == 'basketball':
            base_color = [0.9, 0.4, 0.1, 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 0.8)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif ball_type == 'beach_ball':
            bright_colors = [
                [1.0, 0.2, 0.2, 1.0],
                [0.2, 0.2, 1.0, 1.0],
                [1.0, 1.0, 0.2, 1.0],
                [0.2, 1.0, 0.2, 1.0],
            ]
            base_color = random.choice(bright_colors)
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.2, 0.4)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        else:
            base_color = [random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.7, 0.9)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        voronoi_texture = nodes.new(type='ShaderNodeTexVoronoi')
        voronoi_texture.location = (-400, 100)
        voronoi_texture.inputs['Scale'].default_value = random.uniform(20, 60)
        
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 100)
        color_ramp.color_ramp.elements[0].position = 0.3
        color_ramp.color_ramp.elements[1].position = 0.7
        
        links.new(voronoi_texture.outputs['Distance'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Roughness'])
    
    return mat

def generate_scene():
    """Generates objects within the safe placement area to ensure they stay within ground texture."""
    object_types = ['CUBE', 'CONE', 'SPHERE']
    num_objects = random.randint(5, 20)
    generated_objects = []

    # Calculate safe placement area considering:
    # 1. Ground size (60m x 60m)
    # 2. Maximum object size (0.8m)
    # 3. Safety margin (3m) to ensure objects don't clip ground edges
    safe_placement_radius = (GROUND_SIZE / 2) - 3  # 27m from center
    
    print(f"\nGenerating {num_objects} objects within {safe_placement_radius*2:.1f}m × {safe_placement_radius*2:.1f}m area")
    print(f"Ground texture size: {GROUND_SIZE}m × {GROUND_SIZE}m")

    for i in range(num_objects):
        object_type = random.choice(object_types)

        if object_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(size=1)
        elif object_type == 'CONE':
            obj = create_realistic_traffic_cone(f"CONE_{i}")
            bpy.context.view_layer.update()
            
            obj.location = (random.uniform(-safe_placement_radius, safe_placement_radius),
                            random.uniform(-safe_placement_radius, safe_placement_radius),
                            obj.dimensions.z / 2)
            
            obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
            
            generated_objects.append(obj)
            continue
            
        elif object_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5)

        obj = bpy.context.object
        obj.name = f"{object_type}_{i}"

        width = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        length = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        height = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        
        if object_type == 'CUBE':
            obj.scale = (width, length, height)
        elif object_type == 'SPHERE':
            avg_scale = (width + length + height) / 3
            obj.scale = (avg_scale, avg_scale, avg_scale)

        bpy.context.view_layer.update()

        obj.location = (random.uniform(-safe_placement_radius, safe_placement_radius),
                        random.uniform(-safe_placement_radius, safe_placement_radius),
                        obj.dimensions.z / 2)

        obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))

        mat = create_realistic_material(object_type, f"ObjMat_{i}")
        obj.data.materials.append(mat)
        generated_objects.append(obj)
        
    return generated_objects

def enable_gpu_rendering():
    """Enable GPU rendering for faster performance."""
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'CUDA':
            device.use = True
            print(f"Enabled GPU device: {device.name}")
    
    bpy.context.scene.cycles.device = 'GPU'

def get_bounding_box(obj, camera):
    """Calculates the 2D bounding box of an object from the camera's perspective in pixel coordinates."""
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    x_coords = []
    y_coords = []

    for corner in corners:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, corner)

        if 0.0 <= co_2d.x <= 1.0 and 0.0 <= co_2d.y <= 1.0 and co_2d.z > 0:
            x_coords.append(co_2d.x)
            y_coords.append(co_2d.y)

    if not x_coords or not y_coords:
        return None

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    px_min = x_min * render_size[0]
    px_max = x_max * render_size[0]
    py_min = render_size[1] - (y_max * render_size[1])
    py_max = render_size[1] - (y_min * render_size[1])

    if px_max > px_min and py_max > py_min:
        return [int(px_min), int(py_min), int(px_max), int(py_max)]
    return None

def main():
    """Main function to run the entire generation pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "hdri_envs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "textures"), exist_ok=True)

    # Ensure the bpy_extras.object_utils module is loaded
    try:
        import bpy_extras.object_utils
    except ImportError:
        print("Error: bpy_extras.object_utils module not found. It should be part of Blender.")

    # Enable GPU rendering for faster performance
    try:
        enable_gpu_rendering()
        print("GPU rendering enabled successfully")
    except Exception as e:
        print(f"Could not enable GPU rendering: {e}. Falling back to CPU.")

    # Main loop for dataset generation
    for i in range(NUM_IMAGES):
        print(f"\n{'#'*60}")
        print(f"# Generating scene {i+1}/{NUM_IMAGES}")
        print(f"{'#'*60}")

        # Clean up the scene for a new run
        clear_scene()

        # Setup camera at 30m height
        current_camera = setup_camera()

        # Randomly choose HDRI and ground texture
        random_hdri = random.choice(HDRI_PATHS) if HDRI_PATHS else None
        random_ground_texture = random.choice(GROUND_TEXTURE_PATHS) if GROUND_TEXTURE_PATHS else None

        if random_hdri:
            setup_lighting_and_environment(random_hdri)
            print(f"HDRI loaded: {os.path.basename(random_hdri)}")
        else:
            # Fallback if no HDRI paths are provided (using 15m lighting settings)
            sun_height = random.uniform(18, 25)
            bpy.ops.object.light_add(type='SUN', location=(random.uniform(-10, 10), random.uniform(-10, 10), sun_height))
            sun_light = bpy.context.object
            sun_light.data.energy = random.uniform(1.0, 3.0)
            sun_light.rotation_euler = (math.radians(random.uniform(30, 70)), math.radians(random.uniform(-45, 45)), random.uniform(0, 2*math.pi))
            print("Using default sun lighting (no HDRI provided)")

        if random_ground_texture:
            create_ground_plane(random_ground_texture)
            print(f"Ground texture loaded: {os.path.basename(random_ground_texture)}")
        else:
            # Fallback if no ground textures are provided
            bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, enter_editmode=False, align='WORLD')
            bpy.context.object.name = "Ground"
            print("Using default ground plane (no texture provided)")

        # Generate a new random scene with properly sized objects (30-80cm)
        generated_objects = generate_scene()

        # Set render output properties
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, f"image_{i:04d}")

        # --- Cycles Renderer Settings with GPU optimization ---
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 256  # Increased for better quality with GPU
        bpy.context.scene.cycles.use_denoising = True
        
        # GPU-specific optimizations
        bpy.context.scene.cycles.tile_size = 256  # Larger tiles work better with GPU
        
        print(f"\nRendering scene {i+1}/{NUM_IMAGES}...")
        print(f"  Objects: {len(generated_objects)}")
        print(f"  Camera height: {CAMERA_HEIGHT}m")
        print(f"  Resolution: {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")
        print(f"  Samples: {bpy.context.scene.cycles.samples}")

        # Render the scene to a file
        bpy.ops.render.render(write_still=True)

        # --- Annotation Export ---
        annotations = []
        for obj in generated_objects:
            bbox = get_bounding_box(obj, current_camera)
            if bbox:  # Only save if bounding box is valid and visible
                # Add size information to annotations
                dimensions_cm = [round(dim * 100, 1) for dim in obj.dimensions]  # Convert to cm
                annotations.append({
                    "object": obj.name.split('_')[0],  # e.g., "CUBE"
                    "bbox": bbox,  # [xmin, ymin, xmax, ymax]
                    "dimensions_cm": {
                        "width": dimensions_cm[0],
                        "length": dimensions_cm[1], 
                        "height": dimensions_cm[2]
                    },
                    "location_m": {
                        "x": round(obj.location.x, 2),
                        "y": round(obj.location.y, 2),
                        "z": round(obj.location.z, 2)
                    }
                })

        # Save annotations to a JSON file
        annotation_filepath = os.path.join(OUTPUT_DIR, f"annotations_{i:04d}.json")
        with open(annotation_filepath, "w") as f:
            import json
            json.dump({
                "image_info": {
                    "filename": f"image_{i:04d}.png",
                    "width": bpy.context.scene.render.resolution_x,
                    "height": bpy.context.scene.render.resolution_y,
                    "camera_height_m": CAMERA_HEIGHT,
                    "ground_size_m": GROUND_SIZE,
                    "fov_horizontal_deg": round(math.degrees(2 * math.atan((7.6 / 2) / 4.6)), 2)
                },
                "objects": annotations
            }, f, indent=4)

        print(f"\n✓ Rendered image saved: image_{i:04d}.png")
        print(f"✓ Annotations saved: annotations_{i:04d}.json")
        print(f"✓ Generated {len(annotations)} visible objects with sizes 30-80cm each")
        print(f"{'#'*60}\n")

    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images generated: {NUM_IMAGES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Camera height: {CAMERA_HEIGHT}m")
    print(f"Ground texture size: {GROUND_SIZE}m × {GROUND_SIZE}m")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

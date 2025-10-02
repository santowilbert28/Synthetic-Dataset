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
GROUND_SIZE = 40

# Camera height set to 15 meters (1500cm)
CAMERA_HEIGHT = 15  # 15 meters from ground

# Object size constraints in Blender units (assuming 1 unit = 1 meter)
MIN_OBJECT_SIZE = 0.3  # 30cm in meters
MAX_OBJECT_SIZE = 0.8  # 80cm in meters

HDRI_PATHS = [
    "D:/asunama/synthetic datasets/hdri/bambanani_sunset_4k.exr",
    # Add more HDRI paths here for variety
]

# Paths to ground textures (download these first)
GROUND_TEXTURE_PATHS = [
    "D:/asunama/synthetic datasets/textures/rocky_terrain_02_diff_4k.jpg",
]

def clear_scene():
    """Removes all objects from the scene to start fresh."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera():
    """Sets up a camera for a perfect top-down view at 15 meters height."""
    # Add a camera at 15 meters altitude, directly above the center of the scene
    bpy.ops.object.camera_add(location=(0, 0, CAMERA_HEIGHT))
    camera = bpy.context.object

    # Set the rotation to point straight down along the negative Z-axis
    camera.rotation_euler = (0, 0, 0)

    # Use Orthographic camera for consistent top-down view
    camera.data.type = 'ORTHO'
    # Adjusted ortho scale for 15m height to capture good detail of 30-80cm objects
    camera.data.ortho_scale = 25  # Optimized for object visibility at 15m height

    bpy.context.scene.camera = camera

    # Set render resolution
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.resolution_percentage = 100

def setup_lighting_and_environment(hdri_path):
    """Sets up environment lighting using an HDRI optimized for 15m camera height."""
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

    # Adjusted background strength for 15m camera height
    background.inputs['Strength'].default_value = random.uniform(0.4, 1.2)

    # Add a Sun lamp optimized for 15m camera height
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

    # Create a new material for the ground
    mat = bpy.data.materials.new(name="GroundMaterial")
    ground_obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create shader nodes for the PBR material
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = 0, 0

    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(texture_path)
    texture_node.location = -300, 0

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = 200, 0

    # Link nodes
    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # Randomize texture scale/rotation for more variety
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
    
    # Store the current selection to restore later
    original_selection = [obj for obj in bpy.context.selected_objects]
    original_active = bpy.context.active_object
    
    def create_cone_body():
        # Create a cone mesh (dimensions in cm, so divide by 100 for Blender units)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.20,  # Base radius: 20cm (40cm diameter)
            radius2=0.04,  # Top radius: 4cm (8cm diameter) 
            depth=0.60,    # Height: 60cm
            location=(0, 0, 0.30)  # Position at half height
        )
        
        cone = bpy.context.active_object
        cone.name = f"{cone_name}_Body"
        
        # Enter edit mode to modify the cone
        bpy.context.view_layer.objects.active = cone
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select all and add more geometry for smoothness
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=2)
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return cone

    def create_cone_base():
        bpy.ops.mesh.primitive_cube_add(
            size=0.45,  # Base size: 45cm x 45cm (within your range)
            location=(0, 0, 0.035)  # Adjusted for new cone height
        )
        
        base = bpy.context.active_object
        base.name = f"{cone_name}_Base"
        
        # Scale down the height to make it flat like a base
        base.scale[2] = 0.16  # Make it about 7cm thick
        
        # Apply the scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        return base

    def create_cone_top():
        # Create a small cylinder for the top handle
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16,
            radius=0.015,  # 1.5cm radius (3cm diameter)
            depth=0.04,    # 4cm height
            location=(0, 0, 0.64)  # Positioned at top of 60cm cone + base
        )
        
        top = bpy.context.active_object
        top.name = f"{cone_name}_Top"
        
        return top

    def create_cone_materials():
        # Orange material for main body
        orange_mat = bpy.data.materials.new(name=f"Orange_Cone_{cone_name}")
        orange_mat.use_nodes = True
        
        # Clear existing nodes
        nodes = orange_mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        
        links = orange_mat.node_tree.links
        
        # Material Output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Principled BSDF
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (1.0, 0.3, 0.1, 1.0)  # Orange color
        principled.inputs['Roughness'].default_value = 0.3
        
        # Connect nodes
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Reflective silver material for stripe
        silver_mat = bpy.data.materials.new(name=f"Reflective_Silver_{cone_name}")
        silver_mat.use_nodes = True
        
        # Clear existing nodes
        nodes = silver_mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        
        links = silver_mat.node_tree.links
        
        # Material Output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Principled BSDF
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.85, 1.0)  # Silver color
        principled.inputs['Metallic'].default_value = 0.9
        principled.inputs['Roughness'].default_value = 0.1
        
        # Connect nodes
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        return orange_mat, silver_mat

    def apply_materials_and_stripe(cone_body, orange_mat, silver_mat):
        # Assign orange material to the cone
        if len(cone_body.data.materials) == 0:
            cone_body.data.materials.append(orange_mat)
        
        # Add the silver material to the second slot
        cone_body.data.materials.append(silver_mat)
        
        # Enter edit mode to create the stripe
        bpy.context.view_layer.objects.active = cone_body
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Deselect all
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # Switch to face select mode
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        
        # Get mesh data from edit mode
        bm = bmesh.from_edit_mesh(cone_body.data)
        
        # Deselect all faces first
        for face in bm.faces:
            face.select = False
        
        # Select faces in the middle area for the reflective stripe
        for face in bm.faces:
            face_center = face.calc_center_median()
            # Select faces in the stripe area (adjusted for new cone dimensions)
            if 0.20 < face_center.z < 0.28:
                face.select = True
        
        # Update mesh
        bmesh.update_edit_mesh(cone_body.data)
        
        # Assign the silver material to selected faces
        cone_body.active_material_index = 1
        bpy.ops.object.material_slot_assign()
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

    def add_text_to_cone():
        # Create text object
        bpy.ops.object.text_add(location=(0, -0.3, 0.3))
        text_obj = bpy.context.active_object
        text_obj.name = f"{cone_name}_Text"
        
        # Set text properties
        text_obj.data.body = "TRAFFIC"
        text_obj.data.size = 0.03
        text_obj.data.extrude = 0.002
        
        # Rotate text to face forward
        text_obj.rotation_euler = (math.pi/2, 0, 0)
        
        # Convert to mesh
        bpy.ops.object.convert(target='MESH')
        
        return text_obj

    # Create cone components
    cone_body = create_cone_body()
    cone_base = create_cone_base()
    cone_top = create_cone_top()
    
    # Create materials
    orange_mat, silver_mat = create_cone_materials()
    
    # Apply materials to cone body with stripe
    apply_materials_and_stripe(cone_body, orange_mat, silver_mat)
    
    # Apply orange material to base and top
    cone_base.data.materials.append(orange_mat)
    cone_top.data.materials.append(orange_mat)
    
    # Add text
    text_obj = add_text_to_cone()
    text_obj.data.materials.append(orange_mat)
    
    # Select all cone parts and join them
    # Ensure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Deselect all objects first
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    
    # Select the cone parts
    cone_body.select_set(True)
    cone_base.select_set(True)
    cone_top.select_set(True)
    text_obj.select_set(True)
    
    # Set active object
    bpy.context.view_layer.objects.active = cone_body
    
    # Join objects
    bpy.ops.object.join()
    
    # Rename final object
    final_cone = bpy.context.active_object
    final_cone.name = cone_name
    
    # Add smooth shading
    bpy.ops.object.shade_smooth()
    
    # Restore original selection
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
    
    # Clear default nodes except Principled BSDF and Output
    principled_bsdf = nodes.get("Principled BSDF")
    output_node = nodes.get("Material Output")
    
    # Clear other nodes
    for node in nodes:
        if node not in [principled_bsdf, output_node]:
            nodes.remove(node)
    
    if obj_type == 'CUBE':
        # Create realistic box/crate material (cardboard, wood, or metal)
        material_type = random.choice(['cardboard', 'wood', 'metal', 'plastic'])
        
        if material_type == 'cardboard':
            # Brown cardboard texture
            base_color = [random.uniform(0.4, 0.7), random.uniform(0.25, 0.45), random.uniform(0.1, 0.25), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.7, 0.9)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif material_type == 'wood':
            # Wood crate texture
            wood_colors = [
                [0.4, 0.25, 0.15, 1.0],  # Dark wood
                [0.6, 0.4, 0.25, 1.0],   # Medium wood  
                [0.7, 0.5, 0.35, 1.0]    # Light wood
            ]
            base_color = random.choice(wood_colors)
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 0.8)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif material_type == 'metal':
            # Metal container texture
            base_color = [random.uniform(0.6, 0.8), random.uniform(0.6, 0.8), random.uniform(0.6, 0.8), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.2, 0.5)
            principled_bsdf.inputs['Metallic'].default_value = random.uniform(0.7, 0.9)
            
        else:  # plastic
            # Colorful plastic container
            base_color = [random.uniform(0.2, 0.9), random.uniform(0.2, 0.9), random.uniform(0.2, 0.9), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.4)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        # Add procedural noise texture for surface detail
        noise_texture = nodes.new(type='ShaderNodeTexNoise')
        noise_texture.location = (-400, 0)
        noise_texture.inputs['Scale'].default_value = random.uniform(50, 150)
        noise_texture.inputs['Detail'].default_value = random.uniform(2, 8)
        noise_texture.inputs['Roughness'].default_value = random.uniform(0.4, 0.7)
        
        # Mix noise with base color for realistic surface variation
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 0)
        color_ramp.color_ramp.elements[0].position = 0.4
        color_ramp.color_ramp.elements[1].position = 0.6
        
        links.new(noise_texture.outputs['Color'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Roughness'])
        
    elif obj_type == 'SPHERE':
        # Create realistic ball materials (sports balls, beach balls, etc.)
        ball_type = random.choice(['soccer', 'basketball', 'beach_ball', 'rubber'])
        
        if ball_type == 'soccer':
            # Black and white soccer ball pattern
            base_color = [random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.3, 0.6)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif ball_type == 'basketball':
            # Orange basketball texture
            base_color = [0.9, 0.4, 0.1, 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 0.8)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        elif ball_type == 'beach_ball':
            # Colorful beach ball
            bright_colors = [
                [1.0, 0.2, 0.2, 1.0],  # Red
                [0.2, 0.2, 1.0, 1.0],  # Blue
                [1.0, 1.0, 0.2, 1.0],  # Yellow
                [0.2, 1.0, 0.2, 1.0],  # Green
            ]
            base_color = random.choice(bright_colors)
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.2, 0.4)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        else:  # rubber
            # Dark rubber ball
            base_color = [random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), 1.0]
            principled_bsdf.inputs['Base Color'].default_value = base_color
            principled_bsdf.inputs['Roughness'].default_value = random.uniform(0.7, 0.9)
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            
        # Add procedural texture for ball surface patterns
        voronoi_texture = nodes.new(type='ShaderNodeTexVoronoi')
        voronoi_texture.location = (-400, 100)
        voronoi_texture.inputs['Scale'].default_value = random.uniform(20, 60)
        
        # Use voronoi for subtle surface variation
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.location = (-200, 100)
        color_ramp.color_ramp.elements[0].position = 0.3
        color_ramp.color_ramp.elements[1].position = 0.7
        
        links.new(voronoi_texture.outputs['Distance'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Roughness'])
    
    return mat

def generate_scene():
    """Adds objects with sizes in the range of 30cm to 80cm in each dimension."""
    object_types = ['CUBE', 'CONE', 'SPHERE']
    num_objects = random.randint(5, 20)
    generated_objects = []

    for i in range(num_objects):
        object_type = random.choice(object_types)

        if object_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(size=1)
        elif object_type == 'CONE':
            # Create realistic traffic cone using your custom design
            obj = create_realistic_traffic_cone(f"CONE_{i}")
            
            # Update the scene to apply any transformations
            bpy.context.view_layer.update()
            
            # Place cone exactly on the ground
            placement_area = GROUND_SIZE/2 - 3
            obj.location = (random.uniform(-placement_area, placement_area),
                            random.uniform(-placement_area, placement_area),
                            obj.dimensions.z / 2)
            
            # Only rotate around Z-axis (no tilting)
            obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
            
            generated_objects.append(obj)
            continue  # Skip the general object processing for cones
            
        elif object_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5)

        obj = bpy.context.object
        obj.name = f"{object_type}_{i}"

        # Scale objects to be between 30cm and 80cm in each dimension (for non-cones)
        width = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)   # 30-80cm width
        length = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)  # 30-80cm length  
        height = random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)  # 30-80cm height
        
        if object_type == 'CUBE':
            obj.scale = (width, length, height)
        elif object_type == 'SPHERE':
            # For spheres, use average of dimensions to maintain sphere shape
            avg_scale = (width + length + height) / 3
            obj.scale = (avg_scale, avg_scale, avg_scale)

        # Update the scene to apply the scale before reading dimensions
        bpy.context.view_layer.update()

        # Place object exactly on the ground, with no X or Y rotation
        # Ensure objects don't overlap by using a larger spacing
        placement_area = GROUND_SIZE/2 - 3  # Leave more border space
        obj.location = (random.uniform(-placement_area, placement_area),
                        random.uniform(-placement_area, placement_area),
                        obj.dimensions.z / 2)

        obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))

        # Apply realistic material with textures (for non-cones)
        mat = create_realistic_material(object_type, f"ObjMat_{i}")
        obj.data.materials.append(mat)
        generated_objects.append(obj)
        
    return generated_objects

def enable_gpu_rendering():
    """Enable GPU rendering for faster performance."""
    # Set rendering device to GPU
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPENCL' for AMD
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    
    # Enable all available GPU devices
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'CUDA':  # or 'OPENCL' for AMD
            device.use = True
            print(f"Enabled GPU device: {device.name}")
    
    # Set scene to use GPU
    bpy.context.scene.cycles.device = 'GPU'

def get_bounding_box(obj, camera):
    """
    Calculates the 2D bounding box of an object from the camera's perspective in pixel coordinates.
    This is a robust implementation using `world_to_camera_view`.
    """
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    # Get the 8 corners of the object's bounding box in world space
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    x_coords = []
    y_coords = []

    for corner in corners:
        # Convert world space to camera view (normalized device coordinates 0-1)
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, corner)

        # Check if the point is actually within the camera's view
        if 0.0 <= co_2d.x <= 1.0 and 0.0 <= co_2d.y <= 1.0 and co_2d.z > 0: # co_2d.z > 0 means it's in front of the camera
            x_coords.append(co_2d.x)
            y_coords.append(co_2d.y)

    if not x_coords or not y_coords:
        return None # Object is not visible or entirely outside view

    # Calculate min/max x/y in normalized coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Convert to pixel coordinates
    px_min = x_min * render_size[0]
    px_max = x_max * render_size[0]
    py_min = render_size[1] - (y_max * render_size[1]) # Blender's Y is inverted for screen coords
    py_max = render_size[1] - (y_min * render_size[1])

    # Ensure valid bounding box (e.g., min < max)
    if px_max > px_min and py_max > py_min:
        return [int(px_min), int(py_min), int(px_max), int(py_max)]
    return None

def main():
    """Main function to run the entire generation pipeline."""
    # Ensure output directory and subdirectories for HDRI/textures exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "hdri_envs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "textures"), exist_ok=True)

    # Important: Ensure the 'bpy_extras.object_utils' module is loaded
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
        print(f"Generating scene {i+1}/{NUM_IMAGES}...")

        # Clean up the scene for a new run
        clear_scene()

        # Setup camera at 15m height
        setup_camera()
        current_camera = bpy.context.scene.camera

        # Randomly choose HDRI and ground texture
        random_hdri = random.choice(HDRI_PATHS) if HDRI_PATHS else None
        random_ground_texture = random.choice(GROUND_TEXTURE_PATHS) if GROUND_TEXTURE_PATHS else None

        if random_hdri:
            setup_lighting_and_environment(random_hdri)
        else:
            # Fallback if no HDRI paths are provided
            sun_height = random.uniform(18, 25)
            bpy.ops.object.light_add(type='SUN', location=(random.uniform(-10, 10), random.uniform(-10, 10), sun_height))
            sun_light = bpy.context.object
            sun_light.data.energy = random.uniform(1.0, 3.0)
            sun_light.rotation_euler = (math.radians(random.uniform(30, 70)), math.radians(random.uniform(-45, 45)), random.uniform(0, 2*math.pi))

        if random_ground_texture:
            create_ground_plane(random_ground_texture)
        else:
            # Fallback if no ground textures are provided
            bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, enter_editmode=False, align='WORLD')
            bpy.context.object.name = "Ground"

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
        
        print(f"Rendering scene {i+1} with objects sized 30-80cm at 15m camera height...")

        # Render the scene to a file
        bpy.ops.render.render(write_still=True)

        # --- Annotation Export ---
        annotations = []
        for obj in generated_objects:
            bbox = get_bounding_box(obj, current_camera)
            if bbox: # Only save if bounding box is valid and visible
                # Add size information to annotations
                dimensions_cm = [round(dim * 100, 1) for dim in obj.dimensions]  # Convert to cm
                annotations.append({
                    "object": obj.name.split('_')[0], # e.g., "CUBE"
                    "bbox": bbox, # [xmin, ymin, xmax, ymax]
                    "dimensions_cm": {
                        "width": dimensions_cm[0],
                        "length": dimensions_cm[1], 
                        "height": dimensions_cm[2]
                    }
                })

        # Save annotations to a JSON file
        annotation_filepath = os.path.join(OUTPUT_DIR, f"annotations_{i:04d}.json")
        with open(annotation_filepath, "w") as f:
            import json
            json.dump(annotations, f, indent=4)

        print(f"Rendered image and saved annotations for scene {i+1}/{NUM_IMAGES}")
        print(f"Generated {len(annotations)} objects with sizes 30-80cm each")

if __name__ == "__main__":
    main()

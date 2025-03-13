import bpy

def extract_color_from_obj(filepath):
    """Extract color from the .obj file."""
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("# colour"):
                _, _, r, g, b = line.split()
                return float(r), float(g), float(b)
    return None  # Return None if no color is found

def create_material_with_color(color):
    """Create a material with a Principled BSDF shader using the given color."""
    mat = bpy.data.materials.new(name=f"Material_{color}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_node = nodes.get("Principled BSDF")
    principled_node.inputs['Base Color'].default_value = (*color, 1)  # RGBA
    return mat

def create_emission_material():
    """Create a material with an Emission shader with color black."""
    mat = bpy.data.materials.new(name="Emission_Black")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()  # Clear default nodes

    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Color'].default_value = (0, 0, 0, 1)  # Black RGBA
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    mat.use_backface_culling = True
    return mat

obj_paths = []

for p in obj_paths:
    # Extract color from the .obj file
    color = extract_color_from_obj(p)
    if not color:
        continue  # Skip if no color is found

    # Import mesh
    bpy.ops.import_scene.obj(filepath=p)
    imported_obj = bpy.context.view_layer.objects.selected[0]
    #imported_obj.data.materials.pop(index=0)

    # Create and assign materials
    color_material = create_material_with_color(color)
    emission_material = create_emission_material()
    imported_obj.data.materials.append(color_material)
    imported_obj.data.materials.append(emission_material)

    # Add Solidify modifier
    solidify = imported_obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify.thickness = -0.1
    solidify.offset = -1.0
    solidify.use_rim = True
    solidify.use_flip_normals = True
    solidify.material_offset = 1
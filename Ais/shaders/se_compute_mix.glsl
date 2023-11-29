#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform restrict readonly image2D u_input_image;
layout(rgba32f, binding = 1) uniform image2D u_output_image;

uniform float strength;

void main()
{

    float original_weight = min(1.0 - strength, 1.0);
    float filtered_weight = strength;
    if (strength == 2)
    {
        original_weight = 1;
        filtered_weight = 1;
    }
    if (strength == -2)
    {
        original_weight = 1;
        filtered_weight = -1;
    }
    ivec2 pc = ivec2(gl_GlobalInvocationID.xy);
    vec4 original = imageLoad(u_output_image, pc);
    vec4 filtered = imageLoad(u_input_image, pc);
    imageStore(u_output_image, pc, original * original_weight + filtered * filtered_weight);
}
#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform restrict readonly image2D u_input_image;
layout(rgba32f, binding = 1) uniform restrict writeonly image2D u_output_image;

const int M = 16;
const int N = 2 * M + 1;
uniform int direction;

layout(std430, binding = 0) buffer kernel_coefficients {
   float val[];
} kernel;

void main()
{
    //ivec2 size = imageSize(u_input_image);
    ivec2 pc = ivec2(gl_GlobalInvocationID.xy);

    vec4 pxv = vec4(0.0);
    if (direction == 0)
    {
        for (int i=0;i<N;i++)
        {
            pxv += kernel.val[i] * imageLoad(u_input_image, ivec2(pc.x - M + i, pc.y));
        }
    }
    else
    {
        for (int i=0;i<N;i++)
        {
            pxv += kernel.val[i] * imageLoad(u_input_image, ivec2(pc.x, pc.y - M + i));
        }
    }
	imageStore(u_output_image, pc, pxv);
}
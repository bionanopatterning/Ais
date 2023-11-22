#vertex
#version 420

layout(location = 0) in vec3 pos;

uniform mat4 vpMat;
uniform int override_z;
uniform float override_z_val;
uniform float pixel_size;

void main()
{
    float z_val = override_z == 1 ? override_z_val : pos.z;
    float xy_scale = override_z == 1? pixel_size : 1.0;
    gl_Position = vpMat * vec4(xy_scale * pos.xy, z_val, 1.0);
}

#fragment
#version 420

out vec4 fragColour;

void main()
{
    fragColour = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
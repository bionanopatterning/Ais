#vertex
#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 UV;
out vec2 fUV;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;
uniform int render_3d;
uniform float z_pos;
uniform float pixel_size;

void main()
{
    float z_pos = render_3d == 0 ? position.z : z_pos;
    gl_Position = cameraMatrix * modelMatrix * vec4(position.xy, z_pos, 1.0);
    fUV = UV;
}

#fragment
#version 420

layout(binding = 0) uniform sampler2D image;

out vec4 fragmentColor;
in vec2 fUV;
uniform int shader_blend_code;
uniform float intensity;
uniform float alpha;

void main()
{
    vec2 uv = fUV;
    vec3 colour = intensity * texture(image, uv).rgb;
    if (shader_blend_code == 0)
    {
        fragmentColor = vec4(colour, alpha);
    }
    else if (shader_blend_code == 1)
    {
        fragmentColor = vec4(colour * alpha, 1.0);
    }
    else if (shader_blend_code == 2)
    {
        fragmentColor = vec4(colour.rgb, (colour.r + colour.g + colour.b) * alpha * 5.0);
    }
    else if (shader_blend_code == 3)
    {
        fragmentColor = vec4(colour.rgb, (colour.r + colour.g + colour.b) * alpha / 3.0);
    }
}
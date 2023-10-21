#vertex
#version 420

layout(location = 0) in vec2 pos;

out vec2 uv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = 0.5f + pos / 2.0f;
}

#fragment
#version 420

layout(binding = 0) uniform sampler2D image;

in vec2 uv;

out vec4 fragColour;
uniform float alpha;
uniform float intensity;

void main()
{
    vec3 colour = intensity * texture(image, uv).rgb;
    fragColour = vec4(colour, alpha);
}
#vertex
#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 UV;
out vec2 fUV;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(position, 1.0);
    fUV = UV;
}

#fragment
#version 420

layout(binding = 0) uniform sampler2D image;

out vec4 fragmentColor;
in vec2 fUV;

uniform float alpha;
uniform vec3 colour;
uniform float threshold;

void main()
{
    vec2 uv = fUV;
    float pixelValue = float(texture(image, uv).r > threshold);
    if (pixelValue <= 0)
    {
        discard;
    }
    else
    {
        fragmentColor = vec4(colour, alpha);
    }
}
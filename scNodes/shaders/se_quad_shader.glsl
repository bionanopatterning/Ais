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
uniform float contrastMin;
uniform float contrastMax;

void main()
{
    vec2 uv = fUV;
    float pixelValue = texture(image, uv).r;
    float contrastValue = (pixelValue - contrastMin) / (contrastMax - contrastMin);
    fragmentColor = vec4(contrastValue, contrastValue, contrastValue, alpha);
}
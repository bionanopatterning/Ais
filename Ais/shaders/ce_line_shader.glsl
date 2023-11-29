#vertex
#version 420

layout(location = 0) in vec2 xy;
layout(location = 1) in vec3 rgb;

uniform mat4 cameraMatrix;
out vec3 fcolour;

void main()
{
    gl_Position = cameraMatrix * vec4(xy, 0.0, 1.0);
    fcolour = rgb;
}

#fragment
#version 420

out vec4 fragmentColour;
in vec3 fcolour;

void main()
{
    fragmentColour = vec4(fcolour, 1.0);
}
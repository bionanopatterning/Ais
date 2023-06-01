#vertex
#version 420

layout(location = 0) in vec2 xy;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;
out vec2 fXY;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(xy, 0.0, 1.0);
    vec2 fXY = xy;
}

#fragment
#version 420

out vec4 fragmentColour;
in vec2 fXY;

void main()
{
    fragmentColour = vec4(0.0, 0.0, 0.0, 1.0);
}
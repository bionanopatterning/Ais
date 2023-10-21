#vertex
#version 420

layout(location = 0) in vec2 xy;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;
uniform float z_pos;
out vec2 fXY;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(xy, z_pos, 1.0);
    vec2 fXY = xy;
}

#fragment
#version 420

out vec4 fragmentColour;
in vec2 fXY;
uniform float alpha;

void main()
{
    fragmentColour = vec4(0.0, 0.0, 0.0, alpha);
}
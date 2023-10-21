#vertex
#version 420

layout(location = 0) in vec3 xyz;

uniform mat4 cameraMatrix;

void main()
{
    gl_Position = cameraMatrix * vec4(xyz, 1.0);
}

#fragment
#version 420

out vec4 fragmentColour;

void main()
{
    fragmentColour = vec4(0.0f, 0.0f, 0.0f, 1.0);
}
#vertex
#version 420

layout(location = 0) in vec3 pos;

uniform mat4 vpMat;

void main()
{
    gl_Position = vpMat * vec4(pos, 1.0);
}

#fragment
#version 420

out vec4 fragColour;

void main()
{
    fragColour = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
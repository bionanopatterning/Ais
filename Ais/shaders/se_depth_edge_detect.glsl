#vertex
#version 430 core

layout(location = 0) in vec2 pos;

out vec2 fuv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    fuv = 0.5f + pos / 2.0f;
}

#fragment
#version 430 core

in vec2 fuv;
out vec4 FragColor;

uniform sampler2D depthTexture;
uniform float threshold;
uniform float edge_alpha;
uniform float zmin;
uniform float zmax;

float linear_depth(vec2 uv)
{
    float ndc = texture(depthTexture, uv).r * 2.0f - 1.0f;
    return (2.0 * zmin * zmax) / (zmax + zmin - ndc * (zmax - zmin));;

}

void main()
{
    float dl = linear_depth(fuv + vec2(-1.0, 0.0) / textureSize(depthTexture, 0));
    float dr = linear_depth(fuv + vec2(1.0, 0.0) / textureSize(depthTexture, 0));
    float du = linear_depth(fuv + vec2(0.0, -1.0) / textureSize(depthTexture, 0));
    float dd = linear_depth(fuv + vec2(0.0, 1.0) / textureSize(depthTexture, 0));
    float dx = abs(dr - dl);
    float dy = abs(du - dd);
    float md = max(dx, dy);

    if (md > threshold * 1000.0)
    {
        FragColor = vec4(0.0, 0.0, 0.0, edge_alpha);
    }
    else
    {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }

}
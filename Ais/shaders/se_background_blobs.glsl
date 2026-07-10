#vertex
#version 420

layout(location = 0) in vec2 pos;

out vec2 uv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = 0.5 + pos * 0.5;
}

#fragment
#version 420

#define MAXB 16

in vec2 uv;
out vec4 fragColour;

uniform int uN;
uniform vec2 uRes;          // screen size in px
uniform vec3 uBase;         // papery base colour
uniform float uIntensity;
uniform vec2 uPos[MAXB];    // blob centres in px (top-left origin)
uniform float uRad[MAXB];   // blob radii in px
uniform vec3 uCol[MAXB];    // blob colours

void main()
{
    vec2 frag = vec2(uv.x, 1.0 - uv.y) * uRes;   // top-left origin, matches screen coords
    vec3 col = uBase;
    for (int i = 0; i < uN; i++)
    {
        float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
        float w = exp(-d * d * 2.2) * uIntensity;   // soft gaussian falloff
        col = mix(col, uCol[i], clamp(w, 0.0, 1.0));
    }
    fragColour = vec4(col, 1.0);
}

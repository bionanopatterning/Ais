#vertex
#version 420

layout(location = 0) in vec2 pos;   // unit-quad corner in [-1, 1]^2

uniform vec2 uP0;    // segment endpoints, screen px (top-left origin)
uniform vec2 uP1;
uniform float uRad;  // capsule radius (px)
uniform vec2 uRes;   // canvas size (px)

out vec2 vPx;        // this fragment's screen px (top-left origin)

void main()
{
    vec2 d = uP1 - uP0;
    float len = length(d);
    vec2 axis = (len > 1e-4) ? d / len : vec2(1.0, 0.0);
    vec2 perp = vec2(-axis.y, axis.x);
    vec2 center = 0.5 * (uP0 + uP1);
    float halfLen = 0.5 * len + uRad + 1.0;   // extend for round caps (+1px margin)
    float halfWid = uRad + 1.0;
    vec2 p = center + pos.x * halfLen * axis + pos.y * halfWid * perp;
    vPx = p;
    // screen px (top-left origin) -> clip space (y up)
    gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
}

#fragment
#version 420

in vec2 vPx;
out vec4 fragColour;

uniform vec2 uP0;
uniform vec2 uP1;
uniform float uRad;
uniform vec3 uCol;
uniform float uStrength;

void main()
{
    vec2 pa = vPx - uP0;
    vec2 ba = uP1 - uP0;
    float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0.0, 1.0);
    float dist = length(pa - ba * t);          // distance to the segment
    float soft = max(1.5, uRad * 0.5);         // soft band toward the edge
    float a = smoothstep(uRad, uRad - soft, dist) * uStrength;
    if (a <= 0.001) discard;
    fragColour = vec4(uCol * a, a);            // premultiplied alpha
}

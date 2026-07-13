#vertex
#version 420

layout(location = 0) in vec2 pos;   // fullscreen quad in [-1, 1]^2
out vec2 uv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = 0.5 + pos * 0.5;
}

#fragment
#version 420

in vec2 uv;
out vec4 fragColour;

uniform int uFade;          // 1 = fade helper (write 0; blend multiplies the canvas), 0 = composite
uniform vec3 uBase;         // papery base colour
uniform sampler2D uCanvas;  // the persistent stroke canvas (premultiplied alpha)

void main()
{
    if (uFade == 1) { fragColour = vec4(0.0); return; }
    vec4 c = texture(uCanvas, uv);
    fragColour = vec4(uBase * (1.0 - c.a) + c.rgb, 1.0);   // premultiplied over base
}

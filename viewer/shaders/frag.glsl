#version 450 core

flat in vec3 fcol;

out vec4 color;

void main(void)
{
    color = vec4(fcol, 1.f);
}


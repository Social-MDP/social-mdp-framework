#pragma once

#include "smdp.h"

void renderState(const Context &ctx, const State &s);
void saveCapture(const std::string &path);
void renderLoop();
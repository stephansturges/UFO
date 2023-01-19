# UFO: Uncovering Flights of Oddities
lightweight embeddable DNNs for spotting flying objects

# What does this do? 

Single-shot detection of UAV in realtime on embedded hardware. 
Right now it can only detect Shahed136 suicide drones.

# How can I use this?

You will need any Luxonis device with an RGB camera and the correct version of the depthai-python library installed for your platform and device combination. In terms of real-world use I would recommend that you get a device with a global shutter RGB camera with high light sensitivity and relatively low optical distortion.

# Practical device recommendations:

Any Luxonis device will do to run the code contained here on local images. If you want to deploy this on a camera in the real world: it's up to you to figure out the best device, this is just a proof of concept for now :) 


# Sources:
Some code taken from the excellent https://github.com/luxonis/depthai-experiments from Luxonis.


# Copyright is MIT license
All novel material copyright Stephan Sturges 2022

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# isaac_quad_sim2real

pip install -e .
pip install -e src/third_parties/rsl_rl


## Install toolchain to build and firmware
Using the docker toolbelt
https://www.bitcraze.io/documentation/repository/toolbelt/master/
sudo apt-get install make gcc-arm-none-eabi
cd src/third_parties/
tb make cf2_defconfig
tb make
make cload
# INSTRUCTIONS TO COMPILE THE EXAMPLE ASSUMING THE
# CUDA TOOLKIT IS INSTALLED AT /usr/local/cuda-6.5/
# REMEMBER THAT YOU WILL NEED A KEY LICENSE FILE TO
# RUN THIS EXAMPLE IF YOU ARE USING CUDA 6.5
nvcc -arch=sm_35 -rdc=true -c src/thrust_fft_example.cu
nvcc -arch=sm_35 -dlink -o thrust_fft_example_link.o thrust_fft_example.o -lcudart -lcufft_static
g++ thrust_fft_example.o thrust_fft_example_link.o -L/usr/local/cuda-6.5/lib64 -lcudart -lcufft_static -lculibos

# TO VERIFY THAT THE EXECUTABLE IS NOT USING THE SHARED CUFFT
file a.out
ldd a.out

import mir.ndslice.slice: sliced;
import mir.ndslice.allocation: uninitSlice;
import mir.blas: gemm;

int main()
{
    auto a =
        [-5.0,  1,  7, 7, -4,
           -1, -5,  6, 3, -3,
           -5, -2, -3, 6,  0].sliced(3, 5);

    auto b =
        [-5.0, -3,  3,  1,
            4,  3,  6,  4,
           -4, -2, -2,  2,
           -1,  9,  4,  8,
            9,  8,  3, -2].sliced(5, 4);

   auto c = uninitSlice!double(3, 4);

   // C = 1 * AB + 0 * C
   gemm!double(1, a, b, 0, c);

   if (c !=
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]])
      return 1;

    return 0;
}

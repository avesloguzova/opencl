__kernel void convolution(__global float * fst, __global float * snd,
                          __global float * result, int n, int m)
{
   int row = get_global_id(0);
   int col = get_global_id(1);
   if (row >= n || col >= n)
        return;

   float r = 0.0;
   for (int i = 0; i < m; ++i) {
	   for (int j = 0; j < m; ++j) {
		   int x = row + i - m / 2;
		   int y = col + j - m / 2;

		   if (x >= 0 && x < n && y >= 0 && y < n)
			   r += fst[x * n + y] * snd[i * m + j];
	   }
   }
   result[row * n + col] = r;
}

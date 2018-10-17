#include <stdio.h>

int main() {
  int i = 0;
  int j = 0;
 label:
  for (; i < 100000; i++, j++) {
    printf("%i\n", i);
    if (j % 2 == 0) {
      j++;
      goto label;
    }
  }
  return 0;
}

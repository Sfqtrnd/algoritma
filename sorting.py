import random
import time

# Membuat list angka acak dari 1 sampai 50, dengan 50 elemen
X = [random.randint(1, 50) for _ in range(50)]

# IMPLEMENTASI MERGE SORT
# Pseudocode:
# MergeSort(array)
#   Jika panjang array <= 1:
#       Kembalikan array
#   Bagi array menjadi dua bagian (kiri dan kanan)
#   Panggil MergeSort untuk bagian kiri dan kanan
#   Gabungkan hasil kiri dan kanan menggunakan fungsi Merge

# Merge(left, right)
#   Buat list kosong untuk hasil gabungan
#   Bandingkan elemen pertama dari kedua bagian
#   Tambahkan elemen yang lebih kecil ke dalam list hasil
#   Jika salah satu bagian habis, tambahkan elemen yang tersisa dari bagian lain
#   Kembalikan list hasil gabungan

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# IMPLEMENTASI BUBBLE SORT
# Pseudocode:
# BubbleSort(array)
#   Ulangi untuk setiap elemen dalam array:
#       Tandai apakah ada pertukaran (swapped = False)
#       Bandingkan elemen berdekatan:
#           Jika elemen kiri lebih besar dari elemen kanan, tukar
#           Tandai bahwa ada pertukaran (swapped = True)
#       Jika tidak ada pertukaran, hentikan proses lebih awal
#   Kembalikan array yang sudah diurutkan

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimisasi: Jika tidak ada pertukaran, keluar dari loop
            break
    return arr

# Mengukur waktu eksekusi untuk sorting
# Merge Sort
start_time = time.time()
sorted_merge = merge_sort(X.copy())
merge_sort_time = time.time() - start_time

# Bubble Sort
start_time = time.time()
sorted_bubble = bubble_sort(X.copy())
bubble_sort_time = time.time() - start_time

# Cetak hasil
print("List Awal:", X)
print("\nList yang Diurutkan dengan Merge Sort:", sorted_merge)
print("Waktu Merge Sort:", merge_sort_time, "detik")
print("\nList yang Diurutkan dengan Bubble Sort:", sorted_bubble)
print("Waktu Bubble Sort:", bubble_sort_time, "detik")
var timestamp = '19:09:24.55'; 
var text = '{"Marks": [' + 
'{"Section": "Histogram", "Tests": [' +   
'{"Test": "Test 0", "Output": [' +  
'{"data": {"elapsed_time": 199400, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191982287137900, "id": "a0f45a87-6bec-4a9f-8121-1be2b1d73c12", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191982286938500, "stopped": true}, "id": "a0f45a87-6bec-4a9f-8121-1be2b1d73c12", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "268e4042-c7b4-4735-8b71-a3079a8f35cf", "level": "Trace", "line": 80, "message": "The input length is 16", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982287216100}, "id": "268e4042-c7b4-4735-8b71-a3079a8f35cf", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "e10daf65-7203-40a7-91f8-4824e838d3e2", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982287288400}, "id": "e10daf65-7203-40a7-91f8-4824e838d3e2", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 159538800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191982446846400, "id": "95026f51-339c-4436-8ef9-052bd33f632c", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191982287307600, "stopped": true}, "id": "95026f51-339c-4436-8ef9-052bd33f632c", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 571300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191982447449400, "id": "9202b65e-f267-4ad8-8af3-5ef4c39e1fc8", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191982446878100, "stopped": true}, "id": "9202b65e-f267-4ad8-8af3-5ef4c39e1fc8", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "84fc3209-6a68-4f4c-bfc1-ca0f0da6654a", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982447474100}, "id": "84fc3209-6a68-4f4c-bfc1-ca0f0da6654a", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 128500, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191982447614800, "id": "4483e776-96bf-49cc-9911-880caff06634", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191982447486300, "stopped": true}, "id": "4483e776-96bf-49cc-9911-880caff06634", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 50400, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191982447683300, "id": "4580e928-4bc4-4c42-8771-3b8be49c7f0d", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191982447632900, "stopped": true}, "id": "4580e928-4bc4-4c42-8771-3b8be49c7f0d", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 95900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191982447796900, "id": "0c3d34e5-7e1d-47bc-9a53-4e46d4690975", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191982447701000, "stopped": true}, "id": "0c3d34e5-7e1d-47bc-9a53-4e46d4690975", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/0/myOutput.raw", "expected": "Dataset/Test/0/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 1", "Output": [' +  
'{"data": {"elapsed_time": 2640600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191982614847800, "id": "4243120d-e3d1-4597-97ed-623e50f1946a", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191982612207200, "stopped": true}, "id": "4243120d-e3d1-4597-97ed-623e50f1946a", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "b99906cf-b11c-4e0f-91b6-c4fdac4189ab", "level": "Trace", "line": 80, "message": "The input length is 1024", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982614918800}, "id": "b99906cf-b11c-4e0f-91b6-c4fdac4189ab", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "dab91290-275d-4d4e-875f-78f2f1475a38", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982614999200}, "id": "dab91290-275d-4d4e-875f-78f2f1475a38", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 93422700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191982708442500, "id": "dc69108b-2845-411f-84cb-def0fbfab9ad", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191982615019800, "stopped": true}, "id": "dc69108b-2845-411f-84cb-def0fbfab9ad", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 155700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191982708628800, "id": "0bf595df-cbaa-441d-89e2-531bbb617cc7", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191982708473100, "stopped": true}, "id": "0bf595df-cbaa-441d-89e2-531bbb617cc7", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "346a1898-f892-4c3b-aeb9-1765580ed01c", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982708652400}, "id": "346a1898-f892-4c3b-aeb9-1765580ed01c", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 53800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191982708718200, "id": "be255596-c8a2-4ba4-837d-97920af25d01", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191982708664400, "stopped": true}, "id": "be255596-c8a2-4ba4-837d-97920af25d01", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 42000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191982708777500, "id": "9096c04a-a68b-4301-9027-406eb9c1c9b8", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191982708735500, "stopped": true}, "id": "9096c04a-a68b-4301-9027-406eb9c1c9b8", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 89600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191982708883700, "id": "6028158a-a3a2-429b-8857-38551032440a", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191982708794100, "stopped": true}, "id": "6028158a-a3a2-429b-8857-38551032440a", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/1/myOutput.raw", "expected": "Dataset/Test/1/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 2", "Output": [' +  
'{"data": {"elapsed_time": 1372200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191982863871600, "id": "eb7d8916-b977-4c01-a843-9540881284d1", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191982862499400, "stopped": true}, "id": "eb7d8916-b977-4c01-a843-9540881284d1", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "d1836e51-2982-49c2-94f5-5aacb92ad600", "level": "Trace", "line": 80, "message": "The input length is 513", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982863937100}, "id": "d1836e51-2982-49c2-94f5-5aacb92ad600", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "87eaf5a1-8a59-4c19-a051-5e88144bfced", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982864010800}, "id": "87eaf5a1-8a59-4c19-a051-5e88144bfced", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 90013700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191982954045000, "id": "00a9fb01-9b6b-4dea-9b54-3021e98774b3", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191982864031300, "stopped": true}, "id": "00a9fb01-9b6b-4dea-9b54-3021e98774b3", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 85300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191982954163300, "id": "66e8f646-2b30-425e-ad75-27e1e7426c46", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191982954078000, "stopped": true}, "id": "66e8f646-2b30-425e-ad75-27e1e7426c46", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "70395ca1-9799-4c61-8895-5d582cb53c1a", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191982954186700}, "id": "70395ca1-9799-4c61-8895-5d582cb53c1a", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 53400, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191982954252200, "id": "f0070604-4238-491f-abbc-fe2511340583", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191982954198800, "stopped": true}, "id": "f0070604-4238-491f-abbc-fe2511340583", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 45900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191982954315200, "id": "ffb9d61f-0e06-4783-8665-b4e5faaad157", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191982954269300, "stopped": true}, "id": "ffb9d61f-0e06-4783-8665-b4e5faaad157", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 95600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191982954427700, "id": "0a5f03c1-c27a-4434-93d5-794be543c367", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191982954332100, "stopped": true}, "id": "0a5f03c1-c27a-4434-93d5-794be543c367", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/2/myOutput.raw", "expected": "Dataset/Test/2/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 3", "Output": [' +  
'{"data": {"elapsed_time": 1397600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191983111308700, "id": "7d60dfee-4006-4bea-8c9d-41aaeacc489b", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191983109911100, "stopped": true}, "id": "7d60dfee-4006-4bea-8c9d-41aaeacc489b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "eb3ca82b-fff7-4e19-a9d7-fdc9fd7f89d7", "level": "Trace", "line": 80, "message": "The input length is 511", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983111381500}, "id": "eb3ca82b-fff7-4e19-a9d7-fdc9fd7f89d7", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "5d71717a-0bb6-4de7-986d-19668407f6e4", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983111463400}, "id": "5d71717a-0bb6-4de7-986d-19668407f6e4", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 97235600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191983208720000, "id": "2a8f1083-a183-4a37-99a0-fab437310b0b", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191983111484400, "stopped": true}, "id": "2a8f1083-a183-4a37-99a0-fab437310b0b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 74300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191983208823700, "id": "686f7c4c-2075-48ec-ad2b-9a6dc5737bed", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191983208749400, "stopped": true}, "id": "686f7c4c-2075-48ec-ad2b-9a6dc5737bed", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "aaefb65b-7e3b-4367-b9f8-bcaab1214caa", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983208846700}, "id": "aaefb65b-7e3b-4367-b9f8-bcaab1214caa", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 52500, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191983208911100, "id": "5febe626-9e92-4b11-828c-977509ff9059", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191983208858600, "stopped": true}, "id": "5febe626-9e92-4b11-828c-977509ff9059", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 40300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191983208968800, "id": "3f83266e-2142-4cf2-80d9-fdbf4df145b1", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191983208928500, "stopped": true}, "id": "3f83266e-2142-4cf2-80d9-fdbf4df145b1", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 85300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191983209070500, "id": "4eb97660-120b-489b-932a-a7644c54be68", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191983208985200, "stopped": true}, "id": "4eb97660-120b-489b-932a-a7644c54be68", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/3/myOutput.raw", "expected": "Dataset/Test/3/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 4", "Output": [' +  
'{"data": {"elapsed_time": 149900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191983365415200, "id": "192944ff-da56-4a93-ad65-25070cba7479", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191983365265300, "stopped": true}, "id": "192944ff-da56-4a93-ad65-25070cba7479", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "9f7f2e97-e5b3-4e7d-9a2b-85fcc801d61a", "level": "Trace", "line": 80, "message": "The input length is 1", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983365481900}, "id": "9f7f2e97-e5b3-4e7d-9a2b-85fcc801d61a", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "271a5f9a-3c65-4ded-ac7e-a20336e66280", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983365566600}, "id": "271a5f9a-3c65-4ded-ac7e-a20336e66280", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 92865800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191983458452800, "id": "bc0e98bf-e735-4719-8843-ab7cec2a798d", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191983365587000, "stopped": true}, "id": "bc0e98bf-e735-4719-8843-ab7cec2a798d", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 87100, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191983458569400, "id": "32cddc7b-472b-4651-8c60-979e311388a7", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191983458482300, "stopped": true}, "id": "32cddc7b-472b-4651-8c60-979e311388a7", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "27018ec2-691c-40d7-84e0-f1590fcda04b", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191983458592900}, "id": "27018ec2-691c-40d7-84e0-f1590fcda04b", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 52200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191983458657000, "id": "e0ce9150-2299-4f6b-bd72-5441857118eb", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191983458604800, "stopped": true}, "id": "e0ce9150-2299-4f6b-bd72-5441857118eb", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 40900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191983458715200, "id": "6d639693-73d4-4e58-8dd9-fdc1fb13f03c", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191983458674300, "stopped": true}, "id": "6d639693-73d4-4e58-8dd9-fdc1fb13f03c", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 86300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191983458818100, "id": "88344875-d503-4423-8efb-8992e5c02e2e", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191983458731800, "stopped": true}, "id": "88344875-d503-4423-8efb-8992e5c02e2e", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/4/myOutput.raw", "expected": "Dataset/Test/4/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 5", "Output": [' +  
'{"data": {"elapsed_time": 1090563200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 78, "end_time": 191984718328900, "id": "14a06750-551a-449e-aa91-aee3954c9383", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 74, "start_time": 191983627765700, "stopped": true}, "id": "14a06750-551a-449e-aa91-aee3954c9383", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "92f8355f-4f74-42a4-b1ae-413fc1af4e6b", "level": "Trace", "line": 80, "message": "The input length is 500000", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191984718428500}, "id": "92f8355f-4f74-42a4-b1ae-413fc1af4e6b", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "2eff8e23-3be0-4f2e-aa35-d3c119d970e5", "level": "Trace", "line": 81, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191984718519400}, "id": "2eff8e23-3be0-4f2e-aa35-d3c119d970e5", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 115586000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 93, "end_time": 191984834125700, "id": "5adfecbe-c2cb-4b23-89d2-223955384033", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 191984718539700, "stopped": true}, "id": "5adfecbe-c2cb-4b23-89d2-223955384033", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 366400, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 101, "end_time": 191984834522000, "id": "7601ad9c-7ea8-4e5c-ad5a-34e42adb117d", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 95, "start_time": 191984834155600, "stopped": true}, "id": "7601ad9c-7ea8-4e5c-ad5a-34e42adb117d", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "76c30889-5839-4b47-8243-69c5bba51ec7", "level": "Trace", "line": 105, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 191984834546000}, "id": "76c30889-5839-4b47-8243-69c5bba51ec7", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 3910300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 122, "end_time": 191984838468100, "id": "19ae8358-8361-42ce-b2e2-fa7f54608b0a", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 106, "start_time": 191984834557800, "stopped": true}, "id": "19ae8358-8361-42ce-b2e2-fa7f54608b0a", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 41000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 130, "end_time": 191984838527400, "id": "0909f934-fc6d-43f6-981c-b1ff8603493b", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 124, "start_time": 191984838486400, "stopped": true}, "id": "0909f934-fc6d-43f6-981c-b1ff8603493b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 88000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 137, "end_time": 191984838633200, "id": "3c6d9dd6-bd0d-4367-92c4-874560bcafa7", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 132, "start_time": 191984838545200, "stopped": true}, "id": "3c6d9dd6-bd0d-4367-92c4-874560bcafa7", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": false, "message": "The solution did not match the expected results at row 263. Expecting 108 but got 127."}, "type": "solution"},' + 
'{"data": {"correctq": false, "output": "Dataset/Test/5/myOutput.raw", "expected": "Dataset/Test/5/output.raw"}, "type": "test"}' + 
']}' +  
']}' +  
']}'; 

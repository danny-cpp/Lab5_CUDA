var timestamp = '14:50:13.64'; 
var text = '{"Marks": [' + 
'{"Section": "Histogram", "Tests": [' +   
'{"Test": "Test 0", "Output": [' +  
'{"data": {"elapsed_time": 193500, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176431376315900, "id": "41ad4ff7-7359-4fd7-b056-e4b2b37c06d9", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176431376122400, "stopped": true}, "id": "41ad4ff7-7359-4fd7-b056-e4b2b37c06d9", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "f35a3010-a27c-46d3-90d4-bbf26ec00a59", "level": "Trace", "line": 61, "message": "The input length is 16", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431376391500}, "id": "f35a3010-a27c-46d3-90d4-bbf26ec00a59", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "3bf8fd3d-854c-40dc-9ccb-e21cde3a4cd1", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431376463500}, "id": "3bf8fd3d-854c-40dc-9ccb-e21cde3a4cd1", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 128206700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176431504689200, "id": "c41dc847-0e31-4c10-b8dc-80268f551f2b", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176431376482500, "stopped": true}, "id": "c41dc847-0e31-4c10-b8dc-80268f551f2b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 682800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176431505408000, "id": "efd97461-1902-44dc-9793-e46ff95da86b", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176431504725200, "stopped": true}, "id": "efd97461-1902-44dc-9793-e46ff95da86b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "bbffd356-b803-40b8-bcb1-41c2703359d5", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431505432800}, "id": "bbffd356-b803-40b8-bcb1-41c2703359d5", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 63900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176431505508800, "id": "25c0d9a3-03ec-4e1d-924c-eaab178e233e", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176431505444900, "stopped": true}, "id": "25c0d9a3-03ec-4e1d-924c-eaab178e233e", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 41700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176431505568500, "id": "f4a56f37-e21d-4fe6-86cb-05a53795e695", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176431505526800, "stopped": true}, "id": "f4a56f37-e21d-4fe6-86cb-05a53795e695", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 92700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176431505678100, "id": "0211e849-059b-4c3f-8775-7463873f7d3b", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176431505585400, "stopped": true}, "id": "0211e849-059b-4c3f-8775-7463873f7d3b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/0/myOutput.raw", "expected": "Dataset/Test/0/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 1", "Output": [' +  
'{"data": {"elapsed_time": 2432600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176431658684300, "id": "490a2fe6-e90d-49f9-b3a2-c3616ef6c71c", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176431656251700, "stopped": true}, "id": "490a2fe6-e90d-49f9-b3a2-c3616ef6c71c", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "a5ecac84-0754-4e61-b9fb-26eb71269b48", "level": "Trace", "line": 61, "message": "The input length is 1024", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431658750200}, "id": "a5ecac84-0754-4e61-b9fb-26eb71269b48", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "0ed9852e-f069-416e-89dd-7a3ad9c68b5c", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431658831800}, "id": "0ed9852e-f069-416e-89dd-7a3ad9c68b5c", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 92965800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176431751817400, "id": "49391d8b-cf50-4a03-8936-261c592a65b9", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176431658851600, "stopped": true}, "id": "49391d8b-cf50-4a03-8936-261c592a65b9", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 91100, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176431751939700, "id": "d5570f0e-3661-4085-8270-4832a5223fb8", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176431751848600, "stopped": true}, "id": "d5570f0e-3661-4085-8270-4832a5223fb8", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "7a8b1c04-5d4f-4ac9-aa6f-317650cb2192", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431751973600}, "id": "7a8b1c04-5d4f-4ac9-aa6f-317650cb2192", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 82900, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176431752077600, "id": "72107f4c-7ecb-463a-8e92-e9b3c0d05714", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176431751994700, "stopped": true}, "id": "72107f4c-7ecb-463a-8e92-e9b3c0d05714", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 68500, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176431752175600, "id": "e2332e46-fa4a-4955-b21d-1e9581566ec7", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176431752107100, "stopped": true}, "id": "e2332e46-fa4a-4955-b21d-1e9581566ec7", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 116600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176431752321200, "id": "99082ecd-5dca-49de-a94f-7e196afe9636", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176431752204600, "stopped": true}, "id": "99082ecd-5dca-49de-a94f-7e196afe9636", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/1/myOutput.raw", "expected": "Dataset/Test/1/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 2", "Output": [' +  
'{"data": {"elapsed_time": 1306200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176431899674500, "id": "04579f8e-8ce8-4861-b372-3b10b607453f", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176431898368300, "stopped": true}, "id": "04579f8e-8ce8-4861-b372-3b10b607453f", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "7afe236d-52d9-48cb-b088-177ca3487a09", "level": "Trace", "line": 61, "message": "The input length is 513", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431899738800}, "id": "7afe236d-52d9-48cb-b088-177ca3487a09", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "9912d43d-2568-4e32-96b9-9a539eeaa1a4", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431899815700}, "id": "9912d43d-2568-4e32-96b9-9a539eeaa1a4", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 95451200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176431995287400, "id": "23caf3f3-389d-4d57-b334-e82f2e28a2af", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176431899836200, "stopped": true}, "id": "23caf3f3-389d-4d57-b334-e82f2e28a2af", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 99800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176431995418200, "id": "83572f12-0e33-4b37-9879-e122146b176d", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176431995318400, "stopped": true}, "id": "83572f12-0e33-4b37-9879-e122146b176d", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "5acd9c65-2c15-468a-9a41-ab6bed885a95", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176431995441900}, "id": "5acd9c65-2c15-468a-9a41-ab6bed885a95", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 53300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176431995507300, "id": "c44c7363-fbef-411b-b4aa-fbf2df29c556", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176431995454000, "stopped": true}, "id": "c44c7363-fbef-411b-b4aa-fbf2df29c556", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 41000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176431995565900, "id": "27f8cc2b-b8c3-45b3-9b59-e84ba3b0f959", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176431995524900, "stopped": true}, "id": "27f8cc2b-b8c3-45b3-9b59-e84ba3b0f959", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 88100, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176431995671000, "id": "e54f6f2f-a0bc-4fbd-9dc5-5c22c799d07b", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176431995582900, "stopped": true}, "id": "e54f6f2f-a0bc-4fbd-9dc5-5c22c799d07b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/2/myOutput.raw", "expected": "Dataset/Test/2/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 3", "Output": [' +  
'{"data": {"elapsed_time": 1314000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176432146604800, "id": "8cec6b9c-c9d8-41d4-8b77-5d222e77507e", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176432145290800, "stopped": true}, "id": "8cec6b9c-c9d8-41d4-8b77-5d222e77507e", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "ab515f22-7928-4f33-bca2-87bbe5851b19", "level": "Trace", "line": 61, "message": "The input length is 511", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432146679300}, "id": "ab515f22-7928-4f33-bca2-87bbe5851b19", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "a48f5823-d533-4e25-af8b-449f0d16c5fa", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432146760700}, "id": "a48f5823-d533-4e25-af8b-449f0d16c5fa", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 97114600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176432243896200, "id": "bb4cf2bb-a28d-43f8-8d76-fd984105bcb8", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176432146781600, "stopped": true}, "id": "bb4cf2bb-a28d-43f8-8d76-fd984105bcb8", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 73500, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176432243998800, "id": "ea031792-1039-4ba8-b66f-3a327a2f33b4", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176432243925300, "stopped": true}, "id": "ea031792-1039-4ba8-b66f-3a327a2f33b4", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "e24137ac-ac02-44e9-9288-ea49623495e3", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432244021700}, "id": "e24137ac-ac02-44e9-9288-ea49623495e3", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 52300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176432244086100, "id": "adf5df49-47cf-4125-b002-68437747b7bf", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176432244033800, "stopped": true}, "id": "adf5df49-47cf-4125-b002-68437747b7bf", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 41100, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176432244144700, "id": "d185061f-f8f0-4ce3-b9de-8bf8180606dd", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176432244103600, "stopped": true}, "id": "d185061f-f8f0-4ce3-b9de-8bf8180606dd", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 86000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176432244247500, "id": "3a2a79c1-1351-466f-bd57-c6ad3a01ae8f", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176432244161500, "stopped": true}, "id": "3a2a79c1-1351-466f-bd57-c6ad3a01ae8f", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/3/myOutput.raw", "expected": "Dataset/Test/3/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 4", "Output": [' +  
'{"data": {"elapsed_time": 153800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176432396738500, "id": "b4f513d2-9d23-4b39-b26a-aacf07203b6b", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176432396584700, "stopped": true}, "id": "b4f513d2-9d23-4b39-b26a-aacf07203b6b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "f74c81f4-632a-409c-b1df-bd6be739dbe3", "level": "Trace", "line": 61, "message": "The input length is 1", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432396806400}, "id": "f74c81f4-632a-409c-b1df-bd6be739dbe3", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "7936bc00-52a4-4703-9fe2-7c8b10da7672", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432396881800}, "id": "7936bc00-52a4-4703-9fe2-7c8b10da7672", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 97568000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176432494468700, "id": "fade4de6-dbb3-4dec-8efa-49df2928a2e1", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176432396900700, "stopped": true}, "id": "fade4de6-dbb3-4dec-8efa-49df2928a2e1", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 81000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176432494581400, "id": "24774ea6-de8a-49ca-8e4a-6fffe5eae23e", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176432494500400, "stopped": true}, "id": "24774ea6-de8a-49ca-8e4a-6fffe5eae23e", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "6f833991-4ee0-4aa9-abc3-392922a3dc8e", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176432494605600}, "id": "6f833991-4ee0-4aa9-abc3-392922a3dc8e", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 53300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176432494671300, "id": "6edd5f2f-5e35-43de-b816-647f888caf66", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176432494618000, "stopped": true}, "id": "6edd5f2f-5e35-43de-b816-647f888caf66", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 42700, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176432494732000, "id": "37033a16-2e4d-4fa8-8ac3-c7512922a9c3", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176432494689300, "stopped": true}, "id": "37033a16-2e4d-4fa8-8ac3-c7512922a9c3", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 95400, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176432494844900, "id": "4587724f-55f9-433b-9ce5-8fd2a38a092b", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176432494749500, "stopped": true}, "id": "4587724f-55f9-433b-9ce5-8fd2a38a092b", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/4/myOutput.raw", "expected": "Dataset/Test/4/output.raw"}, "type": "test"}' + 
']},' +  
'{"Test": "Test 5", "Output": [' +  
'{"data": {"elapsed_time": 1016154000, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 59, "end_time": 176433664449600, "id": "41be3032-8d2d-4b9e-a80b-08675c83320c", "idx": 0, "kind": "Generic", "message": "Importing data and creating memory on host", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 55, "start_time": 176432648295600, "stopped": true}, "id": "41be3032-8d2d-4b9e-a80b-08675c83320c", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "3a9dfb12-7bc6-4ab9-a722-74c5fbd19aeb", "level": "Trace", "line": 61, "message": "The input length is 500000", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176433664546500}, "id": "3a9dfb12-7bc6-4ab9-a722-74c5fbd19aeb", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "b3f52c57-47a3-4b60-a6d7-cb3bcf1d0f28", "level": "Trace", "line": 62, "message": "The number of bins is 4096", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176433664627100}, "id": "b3f52c57-47a3-4b60-a6d7-cb3bcf1d0f28", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 87164600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 74, "end_time": 176433751811000, "id": "e54eef06-d8d0-4005-ab10-ebc52a94df46", "idx": 1, "kind": "GPU", "message": "Allocating GPU memory.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 68, "start_time": 176433664646400, "stopped": true}, "id": "e54eef06-d8d0-4005-ab10-ebc52a94df46", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 376600, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 82, "end_time": 176433752218800, "id": "61b14ed0-b589-45cc-8d12-82d7a0d30791", "idx": 2, "kind": "GPU", "message": "Copying input memory to the GPU.", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 76, "start_time": 176433751842200, "stopped": true}, "id": "61b14ed0-b589-45cc-8d12-82d7a0d30791", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "function": "main", "id": "bce99d0f-e6d7-45f4-82e4-1acea00ad9fd", "level": "Trace", "line": 86, "message": "Launching kernel", "mpi_rank": 0, "session_id": "session_id_disabled", "time": 176433752244000}, "id": "bce99d0f-e6d7-45f4-82e4-1acea00ad9fd", "session_id": "session_id_disabled", "type": "logger"},' + 
'{"data": {"elapsed_time": 80300, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 102, "end_time": 176433752336600, "id": "3e227301-b720-4ff8-a34e-e65112c89294", "idx": 3, "kind": "Compute", "message": "Performing CUDA computation", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 87, "start_time": 176433752256300, "stopped": true}, "id": "3e227301-b720-4ff8-a34e-e65112c89294", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 38800, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 110, "end_time": 176433752393400, "id": "addf4c46-d59c-4255-a0aa-ca39201803ca", "idx": 4, "kind": "Copy", "message": "Copying output memory to the CPU", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 104, "start_time": 176433752354600, "stopped": true}, "id": "addf4c46-d59c-4255-a0aa-ca39201803ca", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"elapsed_time": 96200, "end_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "end_function": "main", "end_line": 117, "end_time": 176433752506600, "id": "b478706d-def6-4218-89be-3422a4a553d1", "idx": 5, "kind": "GPU", "message": "Freeing GPU Memory", "mpi_rank": 0, "parent_id": -1, "session_id": "session_id_disabled", "start_file": "D:/Study/DEV/CUDA/Lab5/Lab5/kernel.cu", "start_function": "main", "start_line": 112, "start_time": 176433752410400, "stopped": true}, "id": "b478706d-def6-4218-89be-3422a4a553d1", "session_id": "session_id_disabled", "type": "timer"},' + 
'{"data": {"correctq": true, "message": "The solution is correct"}, "type": "solution"},' + 
'{"data": {"correctq": true, "output": "Dataset/Test/5/myOutput.raw", "expected": "Dataset/Test/5/output.raw"}, "type": "test"}' + 
']}' +  
']}' +  
']}'; 

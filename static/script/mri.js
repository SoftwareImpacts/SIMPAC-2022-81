// no react or anything
let state = {};

// state management
function updateState(newState) {
    state = {...state,
        ...newState
    };
    //console.log(state);
}

function getGradCAM(image_url) {
    var elem = document.createElement('img');
    elem.src = image_url;
    return elem;
}
// event handlers
$("input").change(function(e) {
    let files = document.getElementsByTagName("input")[0].files;
    let filesArr = Array.from(files);
    updateState({
        files: files,
        filesArr: filesArr
    });
    //console.log(files);
    renderFileList();
});

$(".files").on("click", "li > i", function(e) {
    let key = $(this)
        .parent()
        .attr("key");
    let curArr = state.filesArr;
    curArr.splice(key, 1);
    updateState({
        filesArr: curArr
    });
    renderFileList();
});

// $("form").on("submit", function(e) {
//     e.preventDefault();
//     // console.log(state);
//     renderFileList();
//     var fd = new FormData();
//     var ins = document.getElementById('upload').files.length;
//     for (var x = 0; x < ins; x++) {
//         let currentFile = document.getElementById('upload').files[x];
//         fd.append("files[]", currentFile);
//         // console.log(currentFile);
//     }

//     // $.ajax({
//     //         url: '/postmethod',
//     //         type: "POST",
//     //         dataType: 'json',
//     //         data: fd,
//     //         contentType: false,
//     //         cache: false,
//     //         processData: false,

//     //     }).always(function(data) {
//     //         // remove loading image maybe
//     //     })
//     //     .fail(function(data) {
//     //         // handle request failures
//     //     });;
// });

// render functions
function renderFileList() {
    let fileMap = state.filesArr.map((file, index) => {
        let suffix = "bytes";
        let size = file.size;
        if (size >= 1024 && size < 1024000) {
            suffix = "KB";
            size = Math.round(size / 1024 * 100) / 100;
        } else if (size >= 1024000) {
            suffix = "MB";
            size = Math.round(size / 1024000 * 100) / 100;
        }

        return `<li class="upload-items" key="${index}">${
			file.name
		} <span class="file-size">${size} ${suffix}</span><i class="fa-solid fa-trash"></i></li>`;
    });
    $("ul").html(fileMap);
}
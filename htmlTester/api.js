$(document).ready(function () {
    // Define the base URL for your API
    var apiBaseUrl = "http://192.168.39.37:8400";

    // Enroll User Form Submission
    $("#enrollUserForm").submit(function (e) {
        e.preventDefault();
        var user_name = $("#user_name").val();
        $.post(apiBaseUrl + '/enrollUser', { user_name: user_name }, function (data) {
            displayAlert("enrollUserResponse", data);
        });
    });

    // Delete User Form Submission
    $("#deleteUserForm").submit(function (e) {
        e.preventDefault();
        var user_name = $("#delete_user_name").val();
        $.post(apiBaseUrl + '/deleteUser', { user_name: user_name }, function (data) {
            displayAlert("deleteUserResponse", data);
        });
    });

    // Get Users List Button Click
    $("#getUsersButton").click(function () {
        $.post(apiBaseUrl + '/getUsers', function (data) {
            displayUsersList(data);
        });
    });

    // Recognize Face Form Submission
    $("#recognizeFaceForm").submit(function (e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: apiBaseUrl + '/recognizeFace',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (data) {
                displayAlert("recognizeFaceResponse", data);
                console.log(data);
            }
        });
    });

    // Function to display API response in a Bootstrap alert
    function displayAlert(elementId, data) {
		data = JSON.parse(data)
        var alertType = data.error ? "alert-danger" : "alert-success";
        var alertHTML = '<div class="alert ' + alertType + ' mt-3" role="alert">' + data.user_name + '</div>';
        $("#" + elementId).html(alertHTML);
    }

    // Function to display the user list in a Bootstrap alert
    function displayUsersList(data) {
		data = JSON.parse(data)
        if (data.user_list) {
            var userListHTML = '<div class="alert alert-info mt-3" role="alert"><h4 class="alert-heading">User List</h4><ul>';
            data.user_list.forEach(function (user) {
                userListHTML += '<li>' + user + '</li>';
            });
            userListHTML += '</ul></div>';
            $("#getUsersResponse").html(userListHTML);
        } else {
            displayAlert("getUsersResponse", { error: true, message: "No users found." });
        }
    }
});

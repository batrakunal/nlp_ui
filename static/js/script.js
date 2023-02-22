  // Handles user sign up
  $("form[name=signup_form").submit(function(e) {

    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
  
    $.ajax({
      url: "/user/signup",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/signupUser/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });

// Handles change of password form
  $("form[name=changepassword_form").submit(function(e) {

    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
    
    $.ajax({
      url: "/user/changepassword",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/home/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });

  // Handles change of password by the admin
  $("form[name=changepassbyAdmin_form").submit(function(e) {

    var $form = $(this);
    var $error = $form.find(".error");
    
    var data = $form.serialize();
    console.log(data)
    $.ajax({
      url: "/chpassbyadmin/",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        // console.log("success")
        window.location.href = "/userlist";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });

// Handles login form of the user
  $("form[name=login_form").submit(function(e) {
  
    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
  
    $.ajax({
      url: "/user/login",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/home/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });

// Handles login form of the admin
  $("form[name=admin_login_form").submit(function(e) {
  
    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
  
    $.ajax({
      url: "/user/adminlogin",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/signupUser/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });

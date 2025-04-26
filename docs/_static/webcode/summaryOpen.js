$(document).ready(function() {
    $('a[href^="#"]').on('click', function(e) {
      e.preventDefault();
      var target = this.hash,
        $target = $(target);
        $(target + ' details').attr('open', true);

      $('html, body').stop().animate({
        'scrollTop': $target.offset().top
      }, 500, 'swing', function() {
        window.location.hash = target;
      });
  
    });
  });
//https://stackoverflow.com/a/48258026/11028959

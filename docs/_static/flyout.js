$(document).ready(function() {
    var dropdownToggle = $('.sidebar-nav li a.has-dropdown');
    
    dropdownToggle.on('click', function(e) {
      e.preventDefault();
      var thisLink = $(this);
      var thisParent = thisLink.parent();
  
      if (thisParent.hasClass('open')) {
        thisParent.removeClass('open');
      } else {
        $('.sidebar-nav li').removeClass('open'); 
        thisParent.addClass('open');
      }
    });

    $(document).on('click', function(e) {
      if (!$(e.target).closest('.sidebar-nav li').length) {
        $('.sidebar-nav li').removeClass('open');
      }
    });
});
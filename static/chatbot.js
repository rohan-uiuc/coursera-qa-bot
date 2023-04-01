$(document).ready(function() {
    // Initialize variables
    var $chatContainer = $('.chat-container');
    var $chatHeader = $('.chat-header');
    var $chatBody = $('.chat-body');
    var $chatInput = $('.chat-input');
    var $input = $('.chat-input input');
    var $submit = $('.chat_submit');
    var session_id = '';
    $chatBody.children().each(function() {
        $(this).addClass('chat-message');
    });

    // Initialize SocketIO connection
    var socket = io.connect('https://' + document.domain + ':' + location.port);

    // Function to send message to Flask-SocketIO app
    function sendMessage(message) {
        console.log("message: " + message )
        socket.emit('message', {'question': message});
    }

    // Function to display message
    function displayMessage(message, isUser) {
        var $message = $('<div>').addClass('chat-message round');
        var $messageText = $('<p>').html(message.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1">$1</a>'));

        $message.append($messageText);
        if (isUser) {
            $message.addClass('user');
        } else {
            $message.addClass('bot')
        }
        if ($chatBody) {
            $chatBody.append($message);
            if ($chatBody[0]) {
                $chatBody.animate({scrollTop: $chatBody[0].scrollHeight}, 300);
            }
        } else {
            $('.chat-container').append($message);
            $('.chat-container').animate({scrollTop: 0}, 300);
        }
    }


    socket.on('response', function(data) {
        console.log("Received response: " + data.response)
        var response = data.response;
        displayMessage(response, false);
    });


    // Send message on submit
    $submit.click(function(event) {
        event.preventDefault();
        var message = $input.val().trim();
        console.log("Submit clicked: " + message)
        if (message !== '') {
            displayMessage(message, true);
            sendMessage(message);
            $input.val('');
        }
    });

    // Send message on enter key press
    $input.keydown(function(event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            $submit.click();
        }
    });

    // Initial message
    displayMessage('Ask me anything');

    // Function to minimize the widget
    function minimizeWidget() {
        $chatContainer.addClass('minimized');
        $chatHeader.hide();
        $chatBody.hide()
        $chatInput.hide();
        $chatContainer.append('<div class="chat-bot-icon"><i class="fa fa-android"></i></div>');
    }

    // Function to maximize the widget
    function maximizeWidget() {
        $chatContainer.removeClass('minimized');
        $chatBody.show()
        $chatHeader.show();
        $chatInput.show();
        $('.chat-bot-icon').remove();
    }

    // Minimize the widget on click of close button
    $chatHeader.find('.chat-close').click(function() {
        minimizeWidget();
    });

    // Maximize the widget on click of chat-bot-icon
    $chatContainer.on('click', '.chat-bot-icon', function() {
        maximizeWidget();
    });

});

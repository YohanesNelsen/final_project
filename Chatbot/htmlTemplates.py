css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/C4E0BAQGVUZ9-aQ6LxA/company-logo_200_200/0/1663340736017/amplifiglobal_logo?e=2147483647&v=beta&t=NUU89kp7CTZF_EIQo1Qq0nCyp7QLpx5C8e2ySXnAjJY">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2021/07/02/04/48/user-6380868_960_720.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
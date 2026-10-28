output "instance_id" {
  value = aws_instance.this.id
}

output "public_ip" {
  value = aws_instance.this.public_ip
}

output "api_url" {
  value = "http://${aws_instance.this.public_ip}:8000"
}

output "ssh_command" {
  value = "ssh -i ${local_sensitive_file.private_key.filename} ubuntu@${aws_instance.this.public_ip}"
}

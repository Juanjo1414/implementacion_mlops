variable "aws_region" {
  description = "Región de AWS donde se despliega la demo."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Tipo de instancia EC2. t3.micro entra en Free Tier; si el build de Docker falla por falta de RAM, sube a t3.small."
  type        = string
  default     = "t3.micro"
}

variable "my_ip" {
  description = "Tu IP pública en formato CIDR (ej. 200.10.20.30/32) para restringir SSH. Obténla con: curl -s ifconfig.me"
  type        = string
}

variable "repo_url" {
  description = "URL del repositorio Git a clonar en la instancia."
  type        = string
  default     = "https://github.com/Juanjo1414/implementacion_mlops.git"
}

variable "admin_token" {
  description = "Token de administración de la API (protege /data/upload y /retrain)."
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Prefijo usado para nombrar los recursos."
  type        = string
  default     = "housing-demo"
}
